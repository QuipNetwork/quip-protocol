# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU SM budget management.

Optionally monitors external GPU load via NVML to throttle
when sharing the GPU.

Yielding modes:
    yielding=False (default): We own this GPU. Static SM
        budget from gpu_utilization config. No monitoring.
    yielding=True: Yield to other GPU users. NVML daemon
        thread polls utilization and throttles when
        external load is detected.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Any

from GPU.driver_budget import record_throttle

_THROTTLE_SLEEP_S = 0.5


def throttle_if_busy(
    scheduler: Any, *, sleep_fn: Any = time.sleep, sleep_s: float = _THROTTLE_SLEEP_S
) -> None:
    """Pause briefly when the GPU is externally busy (yielding mode).

    No-op when ``scheduler`` is None or yielding is off (``should_throttle()``
    returns False). The streaming samplers call this once before pulling each
    result, restoring the back-off the old inline ``_sample_batch`` did — the
    unified driver path bypasses that wrapper, so the throttle moved into the
    sampler. ``sleep_fn`` is injectable for tests.
    """
    if scheduler is not None and scheduler.should_throttle():
        sleep_fn(sleep_s)
        # QUI-867: this sleep runs while the driver generator is suspended, so
        # it is GPU idle time. Attribute it (no-op unless QUIP_DRIVER_BUDGET=1).
        record_throttle(sleep_s)


def throttled_stream(gen: Any, scheduler: Any) -> Any:
    """Yield from ``gen``, calling ``throttle_if_busy`` before each pull.

    Restores the per-result back-off the inline ``_sample_batch`` did; the
    unified driver path bypasses that, so the throttle lives in the streaming
    samplers. The throttle runs before every ``next()`` (including the first),
    matching the original inline ordering exactly.
    """
    while True:
        throttle_if_busy(scheduler)
        try:
            yield next(gen)
        except StopIteration:
            return


logger = logging.getLogger(__name__)

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


# Module-level globals used only inside the monitor child process.
_NVML_INDEX: int = 0
_NVML_HANDLE = None  # type: ignore[assignment]


def poll_nvml_gpu_util() -> int:
    """Zero-arg NVML utilization poll for the monitor process.

    The spawned monitor re-imports this module and calls nvmlInit() once on
    first use; the device index comes from QUIP_NVML_INDEX (set when the
    scheduler spawns the monitor).

    Returns:
        GPU utilization percentage 0-100, or 0 on error.
    """
    global _NVML_INDEX, _NVML_HANDLE  # noqa: PLW0603
    if _NVML_HANDLE is None:
        pynvml.nvmlInit()
        _NVML_INDEX = int(os.environ.get("QUIP_NVML_INDEX", "0"))
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(_NVML_INDEX)
    return int(pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu)


def _is_mps_active() -> bool:
    """Check if NVIDIA MPS daemon is running.

    MPS creates a pipe directory when the daemon starts.
    The path is $CUDA_MPS_PIPE_DIRECTORY or /tmp/nvidia-mps.
    """
    mps_dir = os.environ.get(
        "CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps",
    )
    return os.path.isdir(mps_dir)


def _is_docker() -> bool:
    """Heuristic: /.dockerenv exists or $container is set."""
    return (
        os.path.exists("/.dockerenv")
        or "container" in os.environ
    )


def configure_mps_thread_limit(
    gpu_utilization_pct: int,
    device_id: int,
    yielding: bool,
) -> bool:
    """Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE before context.

    Must be called before any CUDA API call. Sets the env var
    unconditionally (harmless without MPS). Returns True if
    MPS is detected and will enforce the SM ceiling.

    Args:
        gpu_utilization_pct: SM ceiling percentage (1-100).
        device_id: CUDA device index (for log messages).
        yielding: Whether yielding mode is enabled.
    """
    # Full GPU + no yielding — nothing to configure
    if gpu_utilization_pct >= 100 and not yielding:
        return False

    # Static utilization ceiling: set env var for MPS
    # hardware enforcement (harmless without MPS daemon)
    if gpu_utilization_pct < 100:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = (
            str(gpu_utilization_pct)
        )

    mps_active = _is_mps_active()

    if gpu_utilization_pct < 100 and mps_active:
        logger.info(
            "GPU %d: MPS active — SM ceiling enforced "
            "at %d%%",
            device_id,
            gpu_utilization_pct,
        )
    elif yielding and gpu_utilization_pct >= 100:
        # Yielding without a static ceiling — MPS can't
        # enforce dynamic limits, only software scaling
        logger.info(
            "GPU %d: yielding enabled at 100%% — "
            "NVML adaptive scaling is software-only "
            "(MPS requires a static ceiling to enforce "
            "hardware SM limits)",
            device_id,
        )
    elif _is_docker():
        logger.warning(
            "GPU %d: MPS not active in container — "
            "using software nonce reduction only",
            device_id,
        )
        logger.info(
            "GPU %d: For hardware SM sharing, start MPS "
            "on host and run:\n"
            "    docker run --gpus device=%d --ipc=host"
            " \\\n"
            "      -v /var/run/nvidia-mps:"
            "/var/run/nvidia-mps \\\n"
            "      -e CUDA_MPS_PIPE_DIRECTORY="
            "/var/run/nvidia-mps \\\n"
            "      -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="
            "%d \\\n"
            "      <image>",
            device_id, device_id, gpu_utilization_pct,
        )
        if yielding:
            logger.info(
                "GPU %d: For process-based yielding "
                "without MPS, add --pid=host so NVML "
                "can detect sibling containers:\n"
                "    docker run --gpus device=%d "
                "--pid=host \\\n"
                "      <image>",
                device_id, device_id,
            )
    elif gpu_utilization_pct < 100:
        logger.warning(
            "GPU %d: MPS not active — GPU is "
            "time-sliced. Start MPS for hardware SM "
            "sharing:\n"
            "    export CUDA_MPS_PIPE_DIRECTORY="
            "/var/run/nvidia-mps\n"
            "    nvidia-cuda-mps-control -d",
            device_id,
        )

    return mps_active


class KernelScheduler:
    """Manages GPU SM budget.

    Static SM budget from config with optional NVML-based
    throttling when yielding the GPU to other users.

    Args:
        device_id: CUDA device index for NVML queries.
        device_sms: Total SM count on the device.
        gpu_utilization_pct: Config ceiling (1-100).
        yielding: True = yield to other GPU users
            (NVML throttling). False = static budget.
        poll_interval: Seconds between NVML polls
            (only used when yielding=True).
    """

    def __init__(
        self,
        device_id: int = 0,
        device_sms: int = 0,
        gpu_utilization_pct: int = 100,
        yielding: bool = False,
        poll_interval: float = 5.0,
    ):
        self._device_id = device_id
        self._device_sms = device_sms
        self._gpu_utilization_pct = gpu_utilization_pct
        self._yielding = yielding

        # Static budget (ceiling from config)
        self._static_budget = max(
            1,
            int(device_sms * gpu_utilization_pct / 100),
        )

        # NVML monitor (only when yielding=True)
        self._nvml_handle = None
        self._util_value = mp.get_context("spawn").Value("i", 0)
        self._util_proc = None
        self._util_stop = None
        self._poll_interval = poll_interval

        if yielding:
            self._start_nvml_monitor()

    def _start_nvml_monitor(self) -> None:
        """Start NVML utilization monitor process for yielding mode."""
        if not NVML_AVAILABLE:
            logger.warning(
                "pynvml not installed — yielding mode will "
                "use static budget (no adaptive behavior)"
            )
            return

        try:
            # Initialize NVML in the parent to get a handle that gates
            # should_throttle (it returns False when the handle is None).
            # The monitor child does its own nvmlInit() via poll_nvml_gpu_util().
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self._device_id,
            )
        except Exception as e:
            logger.warning(
                "NVML init failed — yielding mode will use static budget: %s",
                e,
            )
            self._nvml_handle = None
            return

        os.environ["QUIP_NVML_INDEX"] = str(self._device_id)
        from GPU.util_monitor import util_monitor_main
        from shared.proc_util import spawn_worker

        self._util_stop = mp.get_context("spawn").Event()
        self._util_proc = spawn_worker(
            util_monitor_main,
            (self._util_value, self._util_stop, self._poll_interval,
             "GPU.gpu_scheduler:poll_nvml_gpu_util"),
            name=f"nvml-monitor-{self._device_id}",
        )
        logger.info(
            "NVML monitor process started for device %d "
            "(yielding, ceiling=%d%%, poll=%.1fs)",
            self._device_id,
            self._gpu_utilization_pct,
            self._poll_interval,
        )

    def get_sm_budget(self) -> int:
        """SM budget is always static from config.

        Yielding mode only affects throttle behavior,
        not SM budget. NVML reports total GPU utilization
        (including our kernels), so adaptive budgeting
        causes a self-throttle death spiral.

        Returns:
            Number of SMs available (>= 1).
        """
        return self._static_budget

    def should_throttle(self) -> bool:
        """Check if external load warrants a brief pause.

        Only relevant when yielding=True. Returns True when
        observed utilization exceeds 90%.
        """
        if not self._yielding or self._nvml_handle is None:
            return False
        return self._util_value.value > 90

    @property
    def yielding(self) -> bool:
        """Yielding mode (True=yield to others, False=static)."""
        return self._yielding

    def stop(self) -> None:
        """Stop the NVML monitor process."""
        if self._util_proc is not None:
            self._util_stop.set()
            from shared.proc_util import terminate_join
            terminate_join(self._util_proc, 2.0)
