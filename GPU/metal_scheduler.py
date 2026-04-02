# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU core budget + IOKit utilization monitoring for Metal.

Metal equivalent of gpu_scheduler.py. Uses IOKit to query
Apple Silicon GPU utilization via the IOAccelerator service,
and provides a core budget for threadgroup dispatch.

Yielding modes:
    yielding=False (default): Static core budget from
        gpu_utilization config. No monitoring.
    yielding=True: IOKit daemon thread polls GPU utilization
        and triggers throttling when external load is high.
"""

import ctypes
import ctypes.util
import logging
import threading
import time
from typing import Optional


logger = logging.getLogger(__name__)


# ── IOKit GPU utilization query ──────────────────────────

def _query_iokit_gpu_utilization() -> int:
    """Query GPU utilization percentage via IOKit.

    Walks the IOAccelerator service to find
    PerformanceStatistics -> "Device Utilization %".

    Returns:
        GPU utilization 0-100, or 0 on any error.
    """
    try:
        iokit_path = ctypes.util.find_library("IOKit")
        cf_path = ctypes.util.find_library("CoreFoundation")
        if iokit_path is None or cf_path is None:
            return 0
        iokit = ctypes.cdll.LoadLibrary(iokit_path)
        cf = ctypes.cdll.LoadLibrary(cf_path)
    except (OSError, TypeError):
        return 0

    # Type aliases
    kern_return_t = ctypes.c_int
    mach_port_t = ctypes.c_uint
    io_iterator_t = ctypes.c_uint
    io_object_t = ctypes.c_uint
    CFMutableDictionaryRef = ctypes.c_void_p
    CFStringRef = ctypes.c_void_p
    CFTypeRef = ctypes.c_void_p

    # IOServiceMatching
    iokit.IOServiceMatching.restype = CFMutableDictionaryRef
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]

    # IOServiceGetMatchingServices
    iokit.IOServiceGetMatchingServices.restype = kern_return_t
    iokit.IOServiceGetMatchingServices.argtypes = [
        mach_port_t, CFMutableDictionaryRef,
        ctypes.POINTER(io_iterator_t),
    ]

    # IOIteratorNext
    iokit.IOIteratorNext.restype = io_object_t
    iokit.IOIteratorNext.argtypes = [io_iterator_t]

    # IORegistryEntryCreateCFProperties
    iokit.IORegistryEntryCreateCFProperties.restype = kern_return_t
    iokit.IORegistryEntryCreateCFProperties.argtypes = [
        io_object_t,
        ctypes.POINTER(CFMutableDictionaryRef),
        ctypes.c_void_p,  # allocator
        ctypes.c_uint,     # options
    ]

    # IOObjectRelease
    iokit.IOObjectRelease.restype = kern_return_t
    iokit.IOObjectRelease.argtypes = [io_object_t]

    # CoreFoundation helpers
    cf.CFDictionaryGetValue.restype = CFTypeRef
    cf.CFDictionaryGetValue.argtypes = [
        CFTypeRef, CFStringRef,
    ]
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFNumberGetValue.argtypes = [
        CFTypeRef, ctypes.c_int, ctypes.c_void_p,
    ]
    cf.CFRelease.restype = None
    cf.CFRelease.argtypes = [CFTypeRef]

    kCFNumberSInt64Type = 4

    def _cfstr(s: str) -> CFStringRef:
        cf.CFStringCreateWithCString.restype = CFStringRef
        cf.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint,
        ]
        return cf.CFStringCreateWithCString(
            None, s.encode("utf-8"), 0x08000100,
        )

    try:
        matching = iokit.IOServiceMatching(b"IOAccelerator")
        if not matching:
            return 0

        iterator = io_iterator_t()
        # kIOMasterPortDefault = 0
        ret = iokit.IOServiceGetMatchingServices(
            0, matching, ctypes.byref(iterator),
        )
        if ret != 0:
            return 0

        best_util = 0
        while True:
            service = iokit.IOIteratorNext(iterator)
            if not service:
                break

            props = CFMutableDictionaryRef()
            ret = iokit.IORegistryEntryCreateCFProperties(
                service, ctypes.byref(props), None, 0,
            )
            iokit.IOObjectRelease(service)

            if ret != 0 or not props:
                continue

            perf_key = _cfstr("PerformanceStatistics")
            perf_dict = cf.CFDictionaryGetValue(props, perf_key)
            cf.CFRelease(perf_key)

            if perf_dict:
                util_key = _cfstr("Device Utilization %")
                util_val = cf.CFDictionaryGetValue(
                    perf_dict, util_key,
                )
                cf.CFRelease(util_key)

                if util_val:
                    val = ctypes.c_int64(0)
                    if cf.CFNumberGetValue(
                        util_val, kCFNumberSInt64Type,
                        ctypes.byref(val),
                    ):
                        best_util = max(best_util, val.value)

            cf.CFRelease(props)

        iokit.IOObjectRelease(iterator)
        return max(0, min(100, best_util))

    except Exception:
        return 0


class MetalScheduler:
    """GPU core budget + IOKit utilization monitoring for Metal.

    Analogous to KernelScheduler in gpu_scheduler.py but uses
    IOKit instead of NVML, and manages threadgroup counts
    instead of SM counts.

    Args:
        gpu_core_count: Apple Silicon GPU core count.
        gpu_utilization_pct: Config ceiling (1-100).
        yielding: True = yield to other GPU users via IOKit
            monitoring. False = static budget.
        poll_interval: Seconds between IOKit polls (yielding).
    """

    def __init__(
        self,
        gpu_core_count: int,
        gpu_utilization_pct: int = 100,
        yielding: bool = False,
        poll_interval: float = 0.3,
    ):
        self._gpu_core_count = gpu_core_count
        self._gpu_utilization_pct = gpu_utilization_pct
        self._yielding = yielding
        self._poll_interval = poll_interval

        self._static_budget = max(
            1,
            int(gpu_core_count * gpu_utilization_pct / 100),
        )

        # IOKit polling state
        self._external_util_pct = 0
        self._iokit_thread: Optional[threading.Thread] = None
        self._iokit_stop = threading.Event()
        self._util_lock = threading.Lock()
        self._util_cache: Optional[int] = None
        self._util_cache_time = 0.0
        self._CACHE_TTL = 0.3

        # Hysteresis for stable target threadgroups
        self._prev_target = 0
        self._stable_ticks = 0

        if yielding:
            self._start_iokit_monitor()

    def _start_iokit_monitor(self) -> None:
        """Start IOKit polling daemon thread."""
        # Verify IOKit works before starting thread
        test_util = _query_iokit_gpu_utilization()
        if test_util == 0:
            logger.warning(
                "IOKit GPU utilization query returned 0 on "
                "probe — yielding may use static budget only"
            )

        self._iokit_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="MetalUtilMonitor",
        )
        self._iokit_thread.start()
        logger.info(
            "IOKit monitor started (yielding, ceiling=%d%%, "
            "cores=%d, poll=%.1fs)",
            self._gpu_utilization_pct,
            self._gpu_core_count,
            self._poll_interval,
        )

    def _poll_loop(self) -> None:
        """Daemon thread: poll IOKit utilization."""
        while not self._iokit_stop.is_set():
            try:
                util = _query_iokit_gpu_utilization()
                with self._util_lock:
                    self._external_util_pct = util
                    self._util_cache = util
                    self._util_cache_time = time.monotonic()
            except Exception:
                pass  # Keep last known value
            self._iokit_stop.wait(self._poll_interval)

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
        with self._util_lock:
            return self._external_util_pct > 90

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

        with self._util_lock:
            ext_util = self._external_util_pct

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
        with self._util_lock:
            return self._external_util_pct

    def stop(self) -> None:
        """Stop IOKit polling thread."""
        self._iokit_stop.set()
        if self._iokit_thread is not None:
            self._iokit_thread.join(timeout=2.0)


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
