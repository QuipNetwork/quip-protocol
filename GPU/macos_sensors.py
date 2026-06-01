# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""macOS presence/thermal/power/display/GPU sensors for the GPU governor.

Thin ``ctypes`` wrappers over the system frameworks — no new pip dependency
(``pyobjc`` is not required here). Each public sensor returns a plain value and
degrades to a *polite* safe default on any error, never raising into the
governor loop (mirrors ``GPU/util_monitor.py``'s "must never crash the miner"
contract):

    hid_idle_seconds()     -> 0.0      (assume the user is present -> ACTIVE)
    thermal_state()        -> nominal  (don't falsely pause/throttle)
    on_battery()           -> False    (assume AC -> don't pause)
    active_display_count() -> 1        (assume a display -> not headless)
    gpu_active_residency() -> 0        (trim becomes a no-op)

The defaults bias toward responsiveness: a broken sensor must never trap the
machine in PAUSE/LOW, and must never silently keep mining flat-out while the
user is present.

Sources (per the governor design):
    HID idle ← CGEventSourceSecondsSinceLastEventType (CoreGraphics)
    thermal  ← NSProcessInfo.thermalState (Foundation via libobjc)
    power    ← IOPSCopyPowerSourcesInfo / IOPSGetProvidingPowerSourceType (IOKit)
    displays ← CGGetActiveDisplayList (CoreGraphics)
    %GPU     ← IOReport active-residency (primary) │ IOKit (fallback)
"""
from __future__ import annotations

import ctypes
import ctypes.util
from typing import Dict, Optional

# ── thermal-state names (NSProcessInfoThermalState) ─────────────────────
THERMAL_NOMINAL = "nominal"
THERMAL_FAIR = "fair"
THERMAL_SERIOUS = "serious"
THERMAL_CRITICAL = "critical"
THERMAL_STATES = (
    THERMAL_NOMINAL, THERMAL_FAIR, THERMAL_SERIOUS, THERMAL_CRITICAL,
)
_THERMAL_BY_CODE: Dict[int, str] = {
    0: THERMAL_NOMINAL, 1: THERMAL_FAIR, 2: THERMAL_SERIOUS, 3: THERMAL_CRITICAL,
}

# CoreGraphics event-source / display constants.
_kCGEventSourceStateCombinedSessionState = 0
_kCGAnyInputEventType = 0xFFFFFFFF

# CoreFoundation string-encoding for CFStringGetCString.
_kCFStringEncodingUTF8 = 0x08000100

_lib_cache: Dict[str, ctypes.CDLL] = {}


def _load(name: str) -> ctypes.CDLL:
    """Load a system framework/dylib by ``find_library`` name, cached.

    Raises:
        OSError: if the library cannot be located or loaded (the public
            wrappers catch this and return their safe default).
    """
    lib = _lib_cache.get(name)
    if lib is not None:
        return lib
    path = ctypes.util.find_library(name)
    if path is None:
        raise OSError(f"library not found: {name}")
    lib = ctypes.cdll.LoadLibrary(path)
    _lib_cache[name] = lib
    return lib


# ── HID idle ────────────────────────────────────────────────────────────

def _raw_hid_idle_seconds() -> float:
    """Seconds since the last HID input event (CoreGraphics)."""
    cg = _load("CoreGraphics")
    cg.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
    cg.CGEventSourceSecondsSinceLastEventType.argtypes = [
        ctypes.c_int32, ctypes.c_uint32,
    ]
    return float(cg.CGEventSourceSecondsSinceLastEventType(
        _kCGEventSourceStateCombinedSessionState, _kCGAnyInputEventType,
    ))


def hid_idle_seconds() -> float:
    """Return seconds since the last HID input, or 0.0 (present) on error."""
    try:
        return _raw_hid_idle_seconds()
    except Exception:  # noqa: BLE001 — sensor must never raise into the loop
        return 0.0


# ── thermal state ─────────────────────────────────────────────────────────

def _raw_thermal_state() -> int:
    """NSProcessInfo.thermalState as an int (0=nominal … 3=critical)."""
    objc = _load("objc")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    cls = objc.objc_getClass(b"NSProcessInfo")
    if not cls:
        raise OSError("NSProcessInfo class unavailable")

    send = objc.objc_msgSend
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    send.restype = ctypes.c_void_p
    info = send(cls, objc.sel_registerName(b"processInfo"))

    # thermalState returns NSInteger; re-declare restype for this call.
    send.restype = ctypes.c_long
    state = send(info, objc.sel_registerName(b"thermalState"))
    return int(state)


def thermal_state() -> str:
    """Return one of THERMAL_STATES, or nominal on error/unknown code."""
    try:
        return _THERMAL_BY_CODE.get(_raw_thermal_state(), THERMAL_NOMINAL)
    except Exception:  # noqa: BLE001
        return THERMAL_NOMINAL


# ── power source ──────────────────────────────────────────────────────────

def _cfstring_to_str(cf: ctypes.CDLL, cfstr: int) -> str:
    """Copy a CFStringRef into a Python str (empty string if null/failed)."""
    if not cfstr:
        return ""
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
    ]
    buf = ctypes.create_string_buffer(128)
    if cf.CFStringGetCString(cfstr, buf, len(buf), _kCFStringEncodingUTF8):
        return buf.value.decode("utf-8", "replace")
    return ""


def _raw_on_battery() -> bool:
    """True when the providing power source is the battery (IOKit)."""
    iokit = _load("IOKit")
    cf = _load("CoreFoundation")
    iokit.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
    blob = iokit.IOPSCopyPowerSourcesInfo()
    if not blob:
        raise OSError("IOPSCopyPowerSourcesInfo returned null")
    try:
        iokit.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
        iokit.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]
        src = iokit.IOPSGetProvidingPowerSourceType(blob)
        kind = _cfstring_to_str(cf, src)
    finally:
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cf.CFRelease(blob)
    return kind == "Battery Power"


def on_battery() -> bool:
    """Return True when running on battery, or False (AC) on error."""
    try:
        return _raw_on_battery()
    except Exception:  # noqa: BLE001
        return False


# ── active displays (headless detection) ──────────────────────────────────

def _raw_active_display_count() -> int:
    """Number of active displays (CoreGraphics); 0 => headless."""
    cg = _load("CoreGraphics")
    cg.CGGetActiveDisplayList.restype = ctypes.c_int32
    cg.CGGetActiveDisplayList.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
    ]
    count = ctypes.c_uint32(0)
    err = cg.CGGetActiveDisplayList(0, None, ctypes.byref(count))
    if err != 0:
        raise OSError(f"CGGetActiveDisplayList failed: {err}")
    return int(count.value)


def active_display_count() -> int:
    """Return active display count, or 1 (assume attached) on error."""
    try:
        return _raw_active_display_count()
    except Exception:  # noqa: BLE001
        return 1


# ── GPU active residency (closed-loop trim signal) ────────────────────────

def _ioreport_residency() -> Optional[int]:
    """IOReport GPU active-residency probe (0-100), or None if unavailable.

    Seam for the sudoless IOReport channel-sampling path (agtop/macmon-style).
    Until that path is wired it returns None, so ``gpu_active_residency`` uses
    the working IOKit query. The trim is advisory either way (the open-loop
    duty cap is primary), so the IOKit fallback fully preserves behavior.
    """
    return None


def gpu_active_residency() -> int:
    """Return GPU active-residency 0-100 (IOReport primary, IOKit fallback).

    Falls back to the IOKit ``Device Utilization %`` query when IOReport is
    unavailable; returns 0 if both fail (the trim then becomes a no-op).
    """
    try:
        val = _ioreport_residency()
        if val is None:
            val = _query_iokit_gpu_utilization()
        return max(0, min(100, int(val)))
    except Exception:  # noqa: BLE001
        return 0


# ── IOKit "Device Utilization %" query (the measured-load signal) ─────────

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

    except Exception:  # noqa: BLE001 — sensor query must never raise
        return 0


def poll_iokit_gpu_util() -> int:
    """Zero-arg IOKit utilization poll for the monitor process.

    ``_query_iokit_gpu_utilization`` recreates its ctypes handles per call, so
    nothing IOKit-related needs to survive between calls or cross processes.

    Returns:
        GPU utilization percentage 0-100, or 0 on error.
    """
    return int(_query_iokit_gpu_utilization())
