# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Event-based GPU buffer slot lifecycle + SM budget management.

Tracks buffer state via CUDA events for non-blocking kernel
completion detection. Optionally monitors external GPU load
via NVML to reduce SM budget when sharing the GPU.

Yielding modes:
    yielding=False (default): We own this GPU. Static SM
        budget from gpu_utilization config. No monitoring.
    yielding=True: Yield to other GPU users. NVML daemon
        thread polls utilization and reduces SM budget when
        external load is detected.

State machine per slot:
    FREE → UPLOADING → READY → ACTIVE → DONE → FREE
"""

import enum
import logging
import threading
from typing import List, Optional

import cupy as cp


logger = logging.getLogger(__name__)

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class SlotState(enum.Enum):
    """Buffer slot lifecycle states."""
    FREE = "free"
    UPLOADING = "uploading"
    READY = "ready"
    ACTIVE = "active"
    DONE = "done"


class GpuBufferSlot:
    """A single GPU buffer slot with lifecycle tracking.

    Each slot has its own CUDA stream and event for
    non-blocking transfer/completion detection.

    Args:
        slot_id: Unique identifier for this slot.
        stream: CUDA stream for this slot's operations.
        event: CUDA event for completion detection.
    """

    def __init__(
        self,
        slot_id: int,
        stream: cp.cuda.Stream,
        event: cp.cuda.Event,
    ):
        self.slot_id = slot_id
        self.stream = stream
        self.event = event
        self.state = SlotState.FREE
        self.metadata: Optional[dict] = None

    def begin_upload(self) -> None:
        """Transition FREE → UPLOADING."""
        assert self.state == SlotState.FREE, (
            f"Slot {self.slot_id}: begin_upload requires FREE, "
            f"got {self.state}"
        )
        self.state = SlotState.UPLOADING

    def finish_upload(self) -> None:
        """Transition UPLOADING → READY.

        Records event on slot's stream so we can wait for
        transfer completion before kernel launch.
        """
        assert self.state == SlotState.UPLOADING, (
            f"Slot {self.slot_id}: finish_upload requires "
            f"UPLOADING, got {self.state}"
        )
        self.event.record(self.stream)
        self.state = SlotState.READY

    def launch(self, compute_stream: cp.cuda.Stream) -> None:
        """Transition READY → ACTIVE.

        Waits for transfer event, then the caller launches
        the kernel on compute_stream. After kernel launch,
        the event is recorded on compute_stream for
        completion detection.
        """
        assert self.state == SlotState.READY, (
            f"Slot {self.slot_id}: launch requires READY, "
            f"got {self.state}"
        )
        compute_stream.wait_event(self.event)
        self.state = SlotState.ACTIVE

    def record_launch(
        self, compute_stream: cp.cuda.Stream,
    ) -> None:
        """Record completion event after kernel launch."""
        assert self.state == SlotState.ACTIVE, (
            f"Slot {self.slot_id}: record_launch requires "
            f"ACTIVE, got {self.state}"
        )
        self.event.record(compute_stream)

    def harvest(self) -> None:
        """Transition DONE → FREE after results are read."""
        assert self.state == SlotState.DONE, (
            f"Slot {self.slot_id}: harvest requires DONE, "
            f"got {self.state}"
        )
        self.state = SlotState.FREE
        self.metadata = None


class KernelScheduler:
    """Manages GPU buffer slots and SM budget.

    Combines event-based kernel completion detection with
    optional NVML-based SM budget management.

    Args:
        num_slots: Number of buffer slots (typically 2).
        device_id: CUDA device index for NVML queries.
        device_sms: Total SM count on the device.
        gpu_utilization_pct: Config ceiling (1-100).
        yielding: True = yield to other GPU users
            (NVML-adaptive budget). False = static budget.
        poll_interval: Seconds between NVML polls
            (only used when yielding=True).
    """

    def __init__(
        self,
        num_slots: int = 2,
        device_id: int = 0,
        device_sms: int = 0,
        gpu_utilization_pct: int = 100,
        yielding: bool = False,
        poll_interval: float = 5.0,
    ):
        self.slots = [
            GpuBufferSlot(
                slot_id=i,
                stream=cp.cuda.Stream(non_blocking=True),
                event=cp.cuda.Event(),
            )
            for i in range(num_slots)
        ]
        self._compute_stream = cp.cuda.Stream(
            non_blocking=True,
        )

        self._device_id = device_id
        self._device_sms = device_sms
        self._gpu_utilization_pct = gpu_utilization_pct
        self._yielding = yielding

        # Static budget (ceiling from config)
        self._static_budget = max(
            1, int(device_sms * gpu_utilization_pct / 100),
        )

        # NVML monitor (only when yielding=True)
        self._nvml_handle = None
        self._nvml_thread = None
        self._nvml_stop = threading.Event()
        self._external_util_pct = 0
        self._poll_interval = poll_interval

        if yielding:
            self._start_nvml_monitor()

    def _start_nvml_monitor(self) -> None:
        """Start NVML polling thread for yielding mode."""
        if not NVML_AVAILABLE:
            logger.warning(
                "pynvml not installed — yielding mode will "
                "use static budget (no adaptive behavior)"
            )
            return

        try:
            pynvml.nvmlInit()
            self._nvml_handle = (
                pynvml.nvmlDeviceGetHandleByIndex(
                    self._device_id,
                )
            )
            self._nvml_thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
                name=(
                    f"SmBudgetMonitor-{self._device_id}"
                ),
            )
            self._nvml_thread.start()
            logger.info(
                "NVML monitor started for device %d "
                "(yielding, ceiling=%d%%, poll=%.1fs)",
                self._device_id,
                self._gpu_utilization_pct,
                self._poll_interval,
            )
        except Exception as e:
            logger.warning(
                "NVML init failed — yielding mode will "
                "use static budget: %s", e,
            )
            self._nvml_handle = None

    def _poll_loop(self) -> None:
        """Daemon thread: poll NVML utilization."""
        while not self._nvml_stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(
                    self._nvml_handle,
                )
                self._external_util_pct = util.gpu
            except Exception:
                pass  # Keep last known value
            self._nvml_stop.wait(self._poll_interval)

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
        return self._external_util_pct > 90

    @property
    def yielding(self) -> bool:
        """Yielding mode (True=yield to others, False=static)."""
        return self._yielding

    @property
    def compute_stream(self) -> cp.cuda.Stream:
        """Stream used for kernel launches."""
        return self._compute_stream

    def free_slots(self) -> List[GpuBufferSlot]:
        """Return slots in FREE state."""
        return [
            s for s in self.slots if s.state == SlotState.FREE
        ]

    def poll_completed(self) -> List[GpuBufferSlot]:
        """Non-blocking: check events on ACTIVE slots.

        Transitions completed slots to DONE and returns them.
        """
        completed = []
        for slot in self.slots:
            if (slot.state == SlotState.ACTIVE
                    and slot.event.done):
                slot.state = SlotState.DONE
                completed.append(slot)
        return completed

    def any_active(self) -> bool:
        """Return True if any slot is in ACTIVE state."""
        return any(
            s.state == SlotState.ACTIVE for s in self.slots
        )

    def stop(self) -> None:
        """Synchronize streams and stop NVML polling thread."""
        self._compute_stream.synchronize()
        for slot in self.slots:
            slot.stream.synchronize()
        self._nvml_stop.set()
        if self._nvml_thread is not None:
            self._nvml_thread.join(timeout=2.0)
