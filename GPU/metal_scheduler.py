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
    "MetalScheduler", "DutyCycleController", "CapConfig", "Signals",
    "HysteresisState", "decide_tier", "tier_target_pct", "cap_monitor_main",
    "PAUSE", "LOW", "IDLE", "ACTIVE", "LEAVE_POLLS",
]


# ── adaptive cap tier policy (Metal-only, pure) ──────────────────────────

PAUSE = "pause"
LOW = "low"
IDLE = "idle"
ACTIVE = "active"

# Consecutive polls required to *leave* a protective tier (PAUSE/LOW). Entry
# into a protective tier is immediate; only relaxing one is debounced.
LEAVE_POLLS = 2


@dataclass(frozen=True)
class CapConfig:
    """Caps + thresholds the cap policy maps tiers onto.

    Args:
        idle_after_s: Seconds of no HID input before entering IDLE.
        active_util: Cap (%) while the user is present (ACTIVE tier).
        idle_util: Cap (%) when HID-idle or headless (IDLE tier) — the
            existing ``utilization`` key; default 100 keeps headless flat-out.
        serious_util: Cap (%) at thermal == serious (LOW tier).
    """

    idle_after_s: float = 60.0
    active_util: int = 70
    idle_util: int = 100
    serious_util: int = 30


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


def tier_target_pct(tier: str, config: CapConfig) -> int:
    """Map a tier to its ``target_pct`` cap (0-100)."""
    if tier == PAUSE:
        return 0
    if tier == LOW:
        return config.serious_util
    if tier == IDLE:
        return config.idle_util
    return config.active_util


def _read_signals() -> Signals:
    """Read all macOS sensors into a snapshot (each degrades safely)."""
    return Signals(
        idle_s=macos_sensors.hid_idle_seconds(),
        thermal_state=macos_sensors.thermal_state(),
        on_battery=macos_sensors.on_battery(),
        active_displays=macos_sensors.active_display_count(),
    )


def cap_monitor_main(
    target_value,
    measured_value,
    heartbeat_value,
    stop_event,
    poll_interval: float,
    idle_after_s: float,
    active_util: int,
    idle_util: int,
    serious_util: int,
) -> None:
    """Metal adaptive-cap monitor process entry (picklable, spawn-safe).

    Polls the macOS sensors every ``poll_interval``, runs the tier state
    machine, and publishes into three shared ints: the ``target_pct`` cap, the
    measured GPU residency (trim signal), and an incrementing heartbeat for
    staleness detection. Config is passed as plain scalars; sensors are
    resolved in the child. Never raises into the loop (sensors degrade safely).
    """
    config = CapConfig(
        idle_after_s=idle_after_s, active_util=active_util,
        idle_util=idle_util, serious_util=serious_util,
    )
    state = HysteresisState(tier=ACTIVE, leave_count=0)
    if heartbeat_value.value < 0:
        heartbeat_value.value = 0
    while not stop_event.is_set():
        state = decide_tier(_read_signals(), config, state)
        target_value.value = tier_target_pct(state.tier, config)
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
        gpu_utilization_pct: Config ceiling (1-100); also the IDLE/headless cap.
        yielding: True = run the adaptive cap monitor. False = static budget,
            no monitor, flat-out.
        poll_interval: Seconds between sensor polls (yielding).
        active_util: Cap (%) while the user is present (ACTIVE tier).
        idle_after_s: Seconds of no HID input before entering IDLE.
        serious_util: Cap (%) at thermal == serious (LOW tier).
    """

    def __init__(
        self,
        gpu_core_count: int,
        gpu_utilization_pct: int = 100,
        yielding: bool = False,
        poll_interval: float = 0.3,
        active_util: int = 70,
        idle_after_s: float = 60.0,
        serious_util: int = 30,
    ):
        self._gpu_core_count = gpu_core_count
        self._gpu_utilization_pct = gpu_utilization_pct
        self._yielding = yielding
        self._poll_interval = poll_interval
        self._active_util = max(1, min(100, active_util))
        self._idle_after_s = idle_after_s
        self._serious_util = max(1, min(100, serious_util))

        self._static_budget = max(
            1,
            int(gpu_core_count * gpu_utilization_pct / 100),
        )

        # Shared monitor state (published by cap_monitor_main).
        ctx = mp.get_context("spawn")
        self._util_value = ctx.Value("i", 0)          # measured GPU residency
        self._target_value = ctx.Value("i", self._active_util)  # cap %
        self._heartbeat_value = ctx.Value("i", -1)    # ++ each poll
        self._util_proc: Optional[mp.process.BaseProcess] = None
        self._util_stop = None

        # Staleness detection for get_target_pct (fail-safe to active cap).
        self._last_heartbeat = -1
        self._last_hb_time = time.monotonic()
        self._stale_timeout_s = max(2.0, poll_interval * 6)

        # Hysteresis for stable target threadgroups (legacy spatial path;
        # dead under the governor — flagged for dead-code triage).
        self._prev_target = 0
        self._stable_ticks = 0

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
            (self._target_value, self._util_value, self._heartbeat_value,
             self._util_stop, self._poll_interval, self._idle_after_s,
             self._active_util, self._gpu_utilization_pct, self._serious_util),
            name="metal-cap-monitor",
        )
        logger.info(
            "Metal cap monitor started (yielding, idle_cap=%d%%, "
            "active_cap=%d%%, serious_cap=%d%%, idle_after=%.0fs, "
            "cores=%d, poll=%.1fs)",
            self._gpu_utilization_pct, self._active_util, self._serious_util,
            self._idle_after_s, self._gpu_core_count, self._poll_interval,
        )

    def get_target_pct(self) -> int:
        """Return the current cap % (0=pause … 100=flat-out).

        When yielding is off there is no monitor — returns 100 (flat-out).
        While yielding, returns the monitor's published cap, but falls back to
        the ACTIVE cap (never flat-out) if the monitor heartbeat has been
        stale for longer than ``_stale_timeout_s`` (monitor death = fail safe).
        """
        if not self._yielding:
            return 100
        hb = self._heartbeat_value.value
        now = time.monotonic()
        if hb != self._last_heartbeat:
            self._last_heartbeat = hb
            self._last_hb_time = now
        elif now - self._last_hb_time > self._stale_timeout_s:
            return self._active_util
        return self._target_value.value

    def get_measured_gpu(self) -> int:
        """Return the latest measured GPU residency 0-100 (trim signal)."""
        return self._util_value.value

    def get_core_budget(self) -> int:
        """Static budget: gpu_utilization% x core_count.

        Like KernelScheduler.get_sm_budget(), budget is always
        static. Yielding only affects throttle behavior.

        Returns:
            Number of threadgroups (>= 1).
        """
        return self._static_budget

    def should_throttle(self) -> bool:
        """True when external GPU load > 90% (yielding only).

        Mirrors KernelScheduler.should_throttle().
        """
        if not self._yielding:
            return False
        return self._util_value.value > 90

    def compute_target_threadgroups(
        self,
        max_tg: int,
        active_tg: int,
    ) -> int:
        """Target threadgroups based on IOKit utilization.

        Simple fair-share: if external utilization is high,
        reduce dispatch proportionally. Falls back to max_tg
        when yielding is off or IOKit is unavailable.

        Returns:
            Target threadgroup count (>= 1).
        """
        if not self._yielding:
            return max_tg

        ext_util = self._util_value.value

        if ext_util <= 0:
            return max_tg

        # Estimate our contribution
        our_est = (
            self._gpu_utilization_pct
            * active_tg
            / max(max_tg, 1)
        )

        if our_est >= ext_util:
            # Can't distinguish our load — keep current
            return max_tg

        external_load = ext_util - our_est
        target_pct = max(
            self._gpu_utilization_pct / 2,
            self._gpu_utilization_pct - external_load / 2,
        )
        target = round(
            target_pct / self._gpu_utilization_pct * max_tg,
        )
        return max(1, min(target, max_tg))

    def check_stable_target_threadgroups(
        self,
        max_tg: int,
        active_tg: int,
    ) -> Optional[int]:
        """Return target threadgroups only if stable for 2 checks.

        Calls compute_target_threadgroups internally. Returns None
        if the target is still changing between polls (hysteresis
        to prevent stream recreation oscillation).
        """
        current = self.compute_target_threadgroups(max_tg, active_tg)
        if current == self._prev_target:
            self._stable_ticks += 1
        else:
            self._prev_target = current
            self._stable_ticks = 1
        if self._stable_ticks >= 2:
            return current
        return None

    @property
    def yielding(self) -> bool:
        """Whether yielding mode is active."""
        return self._yielding

    def get_cached_utilization(self) -> int:
        """Return latest IOKit GPU utilization without querying.

        Returns:
            Cached utilization 0-100, or 0 if unavailable.
        """
        return self._util_value.value

    def stop(self) -> None:
        """Stop IOKit utilization monitor process."""
        if self._util_proc is not None:
            self._util_stop.set()
            from shared.proc_util import terminate_join
            terminate_join(self._util_proc, 2.0)


class DutyCycleController:
    """Time-based GPU duty cycling for Metal dispatches.

    Measures compute wall-clock time per dispatch and inserts
    proportional sleep to hit a target GPU utilization percentage.
    Uses an exponential moving average to smooth timing and an
    optional IOKit feedback loop to correct drift.

    At 30% target with a 100ms dispatch: sleep = 100 * (1/0.3 - 1)
    = 233ms, creating a real 30/70 compute/idle duty cycle.

    Args:
        target_pct: Target GPU utilization (1-100).
        enabled: Override enable flag. Defaults to target_pct < 100.
    """

    _MIN_SLEEP_S = 0.005   # 5ms floor — ensure GPU scheduler can yield
    _MAX_SLEEP_S = 2.0     # 2s ceiling — keep mining loop responsive
    _EMA_ALPHA = 0.3       # Smoothing factor for compute duration EMA

    def __init__(
        self,
        target_pct: int = 100,
        enabled: Optional[bool] = None,
    ):
        self._target_pct = max(1, min(100, target_pct))
        self._duty_ratio = self._target_pct / 100.0
        self._enabled = (
            enabled if enabled is not None
            else self._target_pct < 100
        )

        # EMA of compute duration (seconds)
        self._ema_compute_s = 0.0
        self._ema_initialized = False

        # PI controller state (Phase 3 feedback)
        self._duty_multiplier = 1.0
        self._kp = 0.01    # Proportional gain
        self._ki = 0.002   # Integral gain
        self._integral = 0.0
        self._integral_clamp = 50.0  # Windup limit

    @property
    def enabled(self) -> bool:
        """Whether duty cycling is active."""
        return self._enabled

    @property
    def target_pct(self) -> int:
        """Target utilization percentage."""
        return self._target_pct

    def compute_sleep(self, compute_duration_s: float) -> float:
        """Compute sleep duration to achieve target duty cycle.

        Args:
            compute_duration_s: Wall-clock time of the GPU dispatch.

        Returns:
            Seconds to sleep before the next dispatch.
        """
        if not self._enabled or compute_duration_s <= 0:
            return self._MIN_SLEEP_S

        # Update EMA
        if not self._ema_initialized:
            self._ema_compute_s = compute_duration_s
            self._ema_initialized = True
        else:
            alpha = self._EMA_ALPHA
            self._ema_compute_s = (
                alpha * compute_duration_s
                + (1.0 - alpha) * self._ema_compute_s
            )

        # Duty cycle formula: sleep = compute * (1/ratio - 1)
        raw_sleep = self._ema_compute_s * (1.0 / self._duty_ratio - 1.0)

        # Apply PI controller multiplier (Phase 3)
        adjusted = raw_sleep * self._duty_multiplier

        return max(self._MIN_SLEEP_S, min(self._MAX_SLEEP_S, adjusted))

    def feedback(self, measured_util_pct: int) -> None:
        """Adjust duty multiplier from IOKit utilization reading.

        PI controller: if measured > target, increase sleep;
        if measured < target, decrease sleep. Call after each
        duty-cycle sleep with the latest IOKit reading.

        Args:
            measured_util_pct: IOKit "Device Utilization %" (0-100).
        """
        if not self._enabled:
            return

        error = measured_util_pct - self._target_pct

        # Accumulate integral with windup clamp
        self._integral = max(
            -self._integral_clamp,
            min(self._integral_clamp, self._integral + error),
        )

        adjustment = self._kp * error + self._ki * self._integral
        self._duty_multiplier = max(
            0.1, min(10.0, self._duty_multiplier + adjustment),
        )

    def reset(self) -> None:
        """Reset EMA and PI state (e.g. after batch size change)."""
        self._ema_compute_s = 0.0
        self._ema_initialized = False
        self._duty_multiplier = 1.0
        self._integral = 0.0

    def set_target(self, target_pct: int) -> None:
        """Retarget the duty cycle at runtime, resetting EMA + PI together.

        Updates ``target_pct`` / duty ratio / enabled flag AND resets the EMA
        and PI integral in lock-step, so a tier change can't carry windup or a
        stale compute-time EMA into the next ``compute_sleep``. A no-op when
        the target is unchanged (preserves accumulated convergence state).

        Args:
            target_pct: New target GPU utilization (1-100). 100 disables
                duty cycling (flat-out monolithic dispatch).
        """
        new_target = max(1, min(100, target_pct))
        if new_target == self._target_pct:
            return
        self._target_pct = new_target
        self._duty_ratio = new_target / 100.0
        self._enabled = new_target < 100
        self.reset()
