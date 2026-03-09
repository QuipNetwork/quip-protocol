# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU utilization monitor with NVML-based adaptive SM limiting.

Polls NVML in a daemon thread to detect external GPU load and
dynamically adjusts the number of SMs available to the miner.
Falls back to static SM limiting if pynvml is not installed.
"""

import logging
import threading

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GpuUtilizationMonitor:
    """Adaptive GPU utilization monitor using NVML.

    Polls GPU utilization rates and computes how many SMs
    the miner should use, respecting both external load and
    the configured ceiling.

    Args:
        device_id: CUDA device index.
        max_utilization_pct: Config ceiling (1-100).
        device_sms: Total SM count on the device.
        poll_interval: Seconds between NVML polls.
    """

    def __init__(
        self,
        device_id: int,
        max_utilization_pct: int,
        device_sms: int,
        poll_interval: float = 5.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.max_utilization_pct = max_utilization_pct
        self.device_sms = device_sms
        self.poll_interval = poll_interval

        # GIL protects single reads/writes of Python ints
        self.external_util_pct = 0
        self._stop_event = threading.Event()
        self._handle = None
        self._thread = None

        if not NVML_AVAILABLE:
            self.logger.warning(
                "pynvml not installed — using static "
                "gpu_utilization=%d%% (no adaptive behavior)",
                max_utilization_pct,
            )
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(
                device_id
            )
            self._thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
                name=f"GpuUtilMonitor-{device_id}",
            )
            self._thread.start()
            self.logger.info(
                "NVML monitor started for device %d "
                "(ceiling=%d%%, poll=%.1fs)",
                device_id, max_utilization_pct, poll_interval,
            )
        except Exception as e:
            self.logger.warning(
                "NVML init failed — using static "
                "gpu_utilization=%d%%: %s",
                max_utilization_pct, e,
            )
            self._handle = None

    def _poll_loop(self):
        """Daemon thread: poll NVML utilization periodically."""
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(
                    self._handle
                )
                self.external_util_pct = util.gpu
            except Exception:
                pass  # Keep last known value
            self._stop_event.wait(self.poll_interval)

    def get_max_sms(self) -> int:
        """Compute available SMs based on external load and ceiling.

        Returns:
            Number of SMs the miner should use (>= 1).
        """
        if self._handle is None:
            # No NVML: static ceiling only
            return max(
                1,
                int(self.device_sms * self.max_utilization_pct / 100),
            )

        reserved_pct = min(self.external_util_pct + 10, 100)
        usable = max(
            1,
            int(self.device_sms * (100 - reserved_pct) / 100),
        )
        ceiling = max(
            1,
            int(self.device_sms * self.max_utilization_pct / 100),
        )
        return min(usable, ceiling)

    def should_throttle(self) -> bool:
        """Check if external load is so high we should sleep.

        Returns True when external utilization exceeds 90%,
        suggesting the miner should add a brief pause between
        jobs to reduce contention.
        """
        if self._handle is None:
            return False
        return self.external_util_pct > 90

    def stop(self):
        """Stop the polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
