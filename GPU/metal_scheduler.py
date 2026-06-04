# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU core budget + adaptive utilization governor for Metal.

Metal equivalent of gpu_scheduler.py, but **entirely independent** of the
CUDA path: it owns its own sensor-driven monitor process (``cap_monitor_main``)
and shares no utilization machinery with ``GPU/util_monitor.py`` (CUDA's NVML
monitor). The only shared code is the generic process lifecycle in
``shared/proc_util.py``.

Yielding modes:
    yielding=False (default): Static core budget from gpu_utilization config.
        No monitor; the sampler runs flat-out monolithic.
    yielding=True: The adaptive cap monitor process polls the macOS sensors
        (HID idle / thermal / battery / displays), runs the tier state machine
        with hysteresis, and publishes a ``target_pct`` cap (0=pause … 100=
        flat-out) plus a measured-GPU trim signal and a heartbeat. The Metal
        sampler re-reads ``target_pct`` per batch to pick its dispatch path.

The IOKit "Device Utilization %" query that used to live here moved to
``GPU/macos_sensors.py`` (its natural home); it is re-exported below so the
dotted path ``GPU.metal_scheduler:poll_iokit_gpu_util`` and existing imports
keep resolving.
"""

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Optional

from GPU import macos_sensors
from GPU.macos_sensors import _query_iokit_gpu_utilization, poll_iokit_gpu_util


logger = logging.getLogger(__name__)

# Re-export so static analysis sees the names as used (back-compat surface).
__all__ = [
    "_query_iokit_gpu_utilization", "poll_iokit_gpu_util",
    "MetalScheduler", "CapConfig", "Signals",
    "HysteresisState", "decide_tier", "tier_budget", "cap_monitor_main",
    "PAUSE", "LOW", "IDLE", "ACTIVE", "LEAVE_POLLS", "UNCAPPED",
    "ACTIVE_DISPATCH_TARGET_MS",
]


# ── adaptive cap tier policy (Metal-only, pure) ──────────────────────────
#
# Apple GPUs jank on TWO levers, and the governor bounds both:
#  1. *Occupancy* — concurrent threads in flight per command buffer
#     (problems x reads). Above a hardware/kernel-specific threshold the
#     execution units / memory bandwidth saturate → the compositor can't
#     interleave → UI stalls. Bounded by the published *thread budget* (the
#     read-split holds each command buffer under it).
#  2. *Duration* — wall-time of a single command buffer. macOS cannot preempt
#     a long-running compute dispatch, so one kernel that runs all of a high
#     ``num_sweeps`` anneal monopolizes the GPU and freezes the UI even when
#     occupancy is capped. Bounded by ``target_dispatch_ms``: while a user is
#     present the sweep schedule is split across short command buffers so the
#     GPU returns to the WindowServer between chunks.

PAUSE = "pause"
LOW = "low"
IDLE = "idle"
ACTIVE = "active"

# Sentinel budget meaning "no cap" (idle/headless → full speed).
UNCAPPED = -1

# Per-command-buffer wall-time target (ms) when a user is present (ACTIVE/LOW).
# Kept well under a 60 Hz frame (16.6 ms) so a chunked anneal yields the GPU
# back to the compositor between dispatches. IDLE/uncapped runs monolithically
# (no UI to protect) for maximum throughput.
ACTIVE_DISPATCH_TARGET_MS = 8.0

# Consecutive polls required to *leave* a protective tier (PAUSE/LOW). Entry
# into a protective tier is immediate; only relaxing one is debounced.
LEAVE_POLLS = 2


@dataclass(frozen=True)
class CapConfig:
    """Thresholds + the occupancy budget the cap policy maps tiers onto.

    Args:
        idle_after_s: Seconds of no HID input before going IDLE (away).
        active_threads: Occupancy budget (max concurrent GPU threads per
            command buffer) while the user is present. ~2048 is smooth on a
            40-core M4 Max with this kernel (≈ 2x maxTotalThreadsPerThreadgroup);
            tune per machine. IDLE/headless runs uncapped; thermal-Serious uses
            half this; battery/critical pauses.
    """

    idle_after_s: float = 60.0
    active_threads: int = 2048


@dataclass(frozen=True)
class Signals:
    """A single poll of the macOS sensors."""

    idle_s: float
    thermal_state: str
    on_battery: bool
    active_displays: int


@dataclass(frozen=True)
class HysteresisState:
    """Effective tier plus the consecutive-poll counter for leaving it."""

    tier: str
    leave_count: int


def _classify(signals: Signals, config: CapConfig) -> str:
    """Raw tier from the priority table (no hysteresis)."""
    if signals.on_battery or signals.thermal_state == macos_sensors.THERMAL_CRITICAL:
        return PAUSE
    if signals.thermal_state == macos_sensors.THERMAL_SERIOUS:
        return LOW
    if signals.active_displays == 0 or signals.idle_s > config.idle_after_s:
        return IDLE
    return ACTIVE


def decide_tier(
    signals: Signals,
    config: CapConfig,
    state: HysteresisState,
) -> HysteresisState:
    """Pure tier decision with asymmetric hysteresis.

    Entering a protective tier (PAUSE/LOW) is immediate; leaving one requires
    ``LEAVE_POLLS`` consecutive polls whose raw classification no longer
    demands it. ACTIVE/IDLE transitions are immediate (IDLE entry is already
    gated by ``idle_after_s``; IDLE exit fires on the first HID event).
    """
    raw = _classify(signals, config)
    prev = state.tier

    if raw == PAUSE:
        return HysteresisState(PAUSE, 0)
    if prev == PAUSE:
        count = state.leave_count + 1
        if count >= LEAVE_POLLS:
            return HysteresisState(raw, 0)
        return HysteresisState(PAUSE, count)

    if raw == LOW:
        return HysteresisState(LOW, 0)
    if prev == LOW:
        count = state.leave_count + 1
        if count >= LEAVE_POLLS:
            return HysteresisState(raw, 0)
        return HysteresisState(LOW, count)

    return HysteresisState(raw, 0)


def tier_budget(tier: str, config: CapConfig) -> int:
    """Map a tier to its occupancy budget (threads/command buffer).

    PAUSE → 0 (stop), LOW → half the active budget, IDLE → UNCAPPED
    (full speed when nobody's present), ACTIVE → ``active_threads``.
    """
    if tier == PAUSE:
        return 0
    if tier == LOW:
        return max(1, config.active_threads // 2)
    if tier == IDLE:
        return UNCAPPED
    return config.active_threads


def _read_signals() -> Signals:
    """Read all macOS sensors into a snapshot (each degrades safely)."""
    return Signals(
        idle_s=macos_sensors.hid_idle_seconds(),
        thermal_state=macos_sensors.thermal_state(),
        on_battery=macos_sensors.on_battery(),
        active_displays=macos_sensors.active_display_count(),
    )


def cap_monitor_main(
    budget_value,
    measured_value,
    heartbeat_value,
    stop_event,
    poll_interval: float,
    idle_after_s: float,
    active_threads: int,
) -> None:
    """Metal adaptive-cap monitor process entry (picklable, spawn-safe).

    Polls the macOS sensors every ``poll_interval``, runs the tier state
    machine, and publishes into three shared ints: the occupancy budget
    (threads/command buffer; 0=pause, -1=uncapped, else cap), the measured GPU
    residency, and an incrementing heartbeat for staleness detection. Config is
    passed as plain scalars; sensors resolve in the child. Never raises into
    the loop (sensors degrade safely).
    """
    config = CapConfig(idle_after_s=idle_after_s, active_threads=active_threads)
    state = HysteresisState(tier=ACTIVE, leave_count=0)
    if heartbeat_value.value < 0:
        heartbeat_value.value = 0
    while not stop_event.is_set():
        state = decide_tier(_read_signals(), config, state)
        budget_value.value = tier_budget(state.tier, config)
        measured_value.value = int(macos_sensors.gpu_active_residency())
        heartbeat_value.value += 1
        stop_event.wait(poll_interval)


class MetalScheduler:
    """GPU core budget + adaptive utilization governor for Metal.

    Analogous to KernelScheduler in gpu_scheduler.py but uses the macOS
    sensors instead of NVML, manages threadgroup counts instead of SM counts,
    and owns its own monitor process (no shared CUDA machinery).

    Args:
        gpu_core_count: Apple Silicon GPU core count.
        gpu_utilization_pct: Problem-batch sizing (core budget = cores x pct%).
        yielding: True = run the adaptive cap monitor. False = no monitor,
            uncapped.
        poll_interval: Seconds between sensor polls (yielding).
        active_threads: Occupancy budget (threads/command buffer) while present.
        idle_after_s: Seconds of no HID input before going IDLE (away).
    """

    def __init__(
        self,
        gpu_core_count: int,
        gpu_utilization_pct: int = 100,
        yielding: bool = False,
        poll_interval: float = 0.3,
        active_threads: int = 2048,
        idle_after_s: float = 60.0,
    ):
        self._gpu_core_count = gpu_core_count
        self._gpu_utilization_pct = gpu_utilization_pct
        self._yielding = yielding
        self._poll_interval = poll_interval
        self._active_threads = max(1, active_threads)
        self._idle_after_s = idle_after_s

        self._static_budget = max(
            1,
            int(gpu_core_count * gpu_utilization_pct / 100),
        )

        # Shared monitor state (published by cap_monitor_main).
        ctx = mp.get_context("spawn")
        self._util_value = ctx.Value("i", 0)             # measured GPU residency
        self._budget_value = ctx.Value("i", self._active_threads)  # thread cap
        self._heartbeat_value = ctx.Value("i", -1)       # ++ each poll
        self._util_proc: Optional[mp.process.BaseProcess] = None
        self._util_stop = None

        # Staleness detection for get_thread_budget (fail-safe to active cap).
        self._last_heartbeat = -1
        self._last_hb_time = time.monotonic()
        self._stale_timeout_s = max(2.0, poll_interval * 6)

        if yielding:
            self._start_cap_monitor()

    def _start_cap_monitor(self) -> None:
        """Spawn the Metal adaptive-cap monitor process (yielding mode).

        Reuses only the generic ``spawn_worker`` lifecycle helper — the poll +
        policy are Metal-specific (``cap_monitor_main``) and share nothing with
        the CUDA util monitor.
        """
        from shared.proc_util import spawn_worker

        self._util_stop = mp.get_context("spawn").Event()
        self._util_proc = spawn_worker(
            cap_monitor_main,
            (self._budget_value, self._util_value, self._heartbeat_value,
             self._util_stop, self._poll_interval, self._idle_after_s,
             self._active_threads),
            name="metal-cap-monitor",
        )
        logger.info(
            "Metal cap monitor started (yielding, active_threads=%d, "
            "idle_after=%.0fs, cores=%d, poll=%.1fs)",
            self._active_threads, self._idle_after_s,
            self._gpu_core_count, self._poll_interval,
        )

    def get_thread_budget(self) -> int:
        """Return the current occupancy budget (threads/command buffer).

        0 = pause, ``UNCAPPED`` (-1) = no cap, else max threads per buffer.
        When yielding is off there is no monitor — returns UNCAPPED. While
        yielding, returns the monitor's published budget, but falls back to the
        ACTIVE budget (never uncapped) if the heartbeat has been stale beyond
        ``_stale_timeout_s`` (monitor death = fail safe, stay polite).
        """
        if not self._yielding:
            return UNCAPPED
        hb = self._heartbeat_value.value
        now = time.monotonic()
        if hb != self._last_heartbeat:
            self._last_heartbeat = hb
            self._last_hb_time = now
        elif now - self._last_hb_time > self._stale_timeout_s:
            return self._active_threads
        return self._budget_value.value

    def target_dispatch_ms(self) -> Optional[float]:
        """Per-command-buffer wall-time budget for sweep-chunking, or None.

        Returns ``None`` when each Metal dispatch may run monolithically (full
        sweeps in one command buffer): yielding off, or the occupancy budget is
        ``UNCAPPED`` (IDLE/headless — no user present, so no UI to protect).
        Returns :data:`ACTIVE_DISPATCH_TARGET_MS` when the governor is capping
        (ACTIVE/LOW — a user is present): the sampler then splits the sweep
        schedule across command buffers held under this wall-time so a long
        anneal can't monopolize the GPU and freeze the compositor. A ``0``
        (PAUSE) budget also returns ``None`` — the streaming loop handles PAUSE
        before dispatching, so chunking never runs in that state.
        """
        if not self._yielding:
            return None
        budget = self.get_thread_budget()
        if budget == UNCAPPED or budget <= 0:
            return None
        return ACTIVE_DISPATCH_TARGET_MS

    def get_measured_gpu(self) -> int:
        """Return the latest measured GPU residency 0-100."""
        return self._util_value.value

    def get_core_budget(self) -> int:
        """Static budget: gpu_utilization% x core_count.

        Like KernelScheduler.get_sm_budget(), budget is always
        static. Yielding only affects throttle behavior.

        Returns:
            Number of threadgroups (>= 1).
        """
        return self._static_budget

    @property
    def yielding(self) -> bool:
        """Whether yielding mode is active."""
        return self._yielding

    def stop(self) -> None:
        """Stop IOKit utilization monitor process."""
        if self._util_proc is not None:
            self._util_stop.set()
            from shared.proc_util import terminate_join
            terminate_join(self._util_proc, 2.0)
