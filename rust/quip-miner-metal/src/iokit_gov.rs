//! IOKit / GPU-util governor (port of `nvml_gov` public interface).
//!
//! Static util ceiling from config; optional background poll when yielding.
//! When yielding and observed util > 90%, the session loop inserts a brief
//! pause so sibling GPU users get time slices.
//!
//! # Sensor source
//!
//! Reads Apple GPU device utilization via IOKit: walks every `IOAccelerator`
//! service, reads its `PerformanceStatistics` CFDictionary, and takes the max
//! `"Device Utilization %"` value across services (a Mac normally has one).
//! This is a direct Rust port of the `ctypes` IOKit binding in
//! `GPU/macos_sensors.py::_query_iokit_gpu_utilization` — same service name,
//! same dictionary keys, same "return 0 on any failure" contract. Keeps the
//! **identical public interface** as `nvml_gov::UtilGovernor` so `session.rs`
//! calls it unchanged.
//!
//! This crate does not use the private/undocumented IOReport channel-sampling
//! path (`GPU/macos_sensors.py`'s `_ioreport_residency`, which is a stub
//! there too) — only the documented IOKit `IOAccelerator` service query.

use core_foundation_sys::base::{CFGetTypeID, CFRelease, CFTypeRef};
use core_foundation_sys::dictionary::{
    CFDictionaryGetTypeID, CFDictionaryGetValue, CFDictionaryRef, CFMutableDictionaryRef,
};
use core_foundation_sys::number::{
    kCFNumberSInt64Type, CFNumberGetTypeID, CFNumberGetValue, CFNumberRef,
};
use core_foundation_sys::string::{kCFStringEncodingUTF8, CFStringCreateWithCString, CFStringRef};
use io_kit_sys::types::io_iterator_t;
use io_kit_sys::{
    kIOMasterPortDefault, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties,
    IOServiceGetMatchingServices, IOServiceMatching,
};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Reconfigurable governor knobs plus the latest util sample, shared with the
/// poll thread.
struct Knobs {
    /// Util ceiling 1–100; throttle fires above it when yielding.
    ceiling: AtomicU32,
    yielding: AtomicBool,
    /// Last GPU util percent 0–100 (0 while not yielding).
    last_util: AtomicU32,
    stop: AtomicBool,
}

/// Shared utilization sample and reconfigurable governor knobs.
pub struct UtilGovernor {
    knobs: Arc<Knobs>,
    handle: Option<JoinHandle<()>>,
}

impl UtilGovernor {
    /// Start the util poller. Values come from the CLI; `Configure` may later
    /// override them via [`reconfigure`](Self::reconfigure). The poll thread
    /// runs regardless of `yielding` (so a later `false -> true` override starts
    /// sampling with no thread churn) but only records util while yielding.
    ///
    /// Falls back to a silent no-op if sensors are unavailable (miner still
    /// runs; util stays 0 and throttle never fires).
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let knobs_thread = Arc::clone(&knobs);
        let handle = Some(thread::spawn(move || {
            poll_loop(device_index, &knobs_thread)
        }));
        Self { knobs, handle }
    }

    /// Override the ceiling and yielding flag at runtime (config over CLI).
    pub fn reconfigure(&self, utilization_ceiling: u32, yielding: bool) {
        self.knobs
            .ceiling
            .store(utilization_ceiling.clamp(1, 100), Ordering::Relaxed);
        self.knobs.yielding.store(yielding, Ordering::Relaxed);
    }

    /// Current ceiling (CLI value, or the config override once applied).
    pub fn utilization_ceiling(&self) -> u32 {
        self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Current yielding flag (CLI value, or the config override once applied).
    pub fn yielding(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
    }

    /// Last GPU util percent (0–100), or 0 if not yielding / unavailable.
    pub fn utilization(&self) -> f32 {
        self.knobs.last_util.load(Ordering::Relaxed) as f32
    }

    /// True when yielding and the last util sample exceeds the ceiling.
    pub fn should_throttle(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
            && self.knobs.last_util.load(Ordering::Relaxed)
                > self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Request the poller to exit and join it.
    pub fn stop(&mut self) {
        self.knobs.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for UtilGovernor {
    fn drop(&mut self) {
        self.stop();
    }
}

// `device_index` is unused: IOKit's `IOAccelerator` matching walks every GPU
// service on the host (there is normally exactly one on Apple Silicon) and
// has no per-index selector analogous to NVML's `device_by_index`.
fn poll_loop(_device_index: u32, knobs: &Knobs) {
    while !knobs.stop.load(Ordering::Relaxed) {
        if knobs.yielding.load(Ordering::Relaxed) {
            knobs
                .last_util
                .store(query_iokit_gpu_utilization(), Ordering::Relaxed);
        } else {
            knobs.last_util.store(0, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_secs(2));
    }
}

/// Wrap a UTF-8 Rust string as a `CFStringRef`, or `None` on any failure.
fn cfstr(s: &str) -> Option<CFStringRef> {
    let c = CString::new(s).ok()?;
    let r =
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), kCFStringEncodingUTF8) };
    if r.is_null() {
        None
    } else {
        Some(r)
    }
}

/// GPU utilization percent (0-100) via IOKit, or 0 on any error.
///
/// Walks the `IOAccelerator` service(s), reading
/// `PerformanceStatistics -> "Device Utilization %"` from the IORegistry.
/// Never panics; a query failure (missing service, unsupported key, ...)
/// degrades to 0, matching the Python reference's `except Exception: return 0`.
fn query_iokit_gpu_utilization() -> u32 {
    unsafe {
        let Ok(service_name) = CString::new("IOAccelerator") else {
            return 0;
        };
        let matching = IOServiceMatching(service_name.as_ptr());
        if matching.is_null() {
            return 0;
        }

        let mut iterator: io_iterator_t = 0;
        let ret = IOServiceGetMatchingServices(
            kIOMasterPortDefault,
            matching as CFDictionaryRef,
            &mut iterator,
        );
        if ret != 0 {
            return 0;
        }

        let mut best: i64 = 0;
        loop {
            let service = IOIteratorNext(iterator);
            if service == 0 {
                break;
            }
            let mut props: CFMutableDictionaryRef = std::ptr::null_mut();
            let pret = IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0);
            IOObjectRelease(service);
            if pret != 0 || props.is_null() {
                continue;
            }

            if let Some(util) = read_device_utilization(props as CFDictionaryRef) {
                best = best.max(util);
            }
            CFRelease(props as CFTypeRef);
        }
        IOObjectRelease(iterator);
        best.clamp(0, 100) as u32
    }
}

/// Best-effort Apple GPU core count via IOKit, or `None` on any failure.
///
/// Reads the `"gpu-core-count"` integer property published on the
/// `IOAccelerator` service (Apple Silicon). Used only to size the streaming
/// command-buffer budget ([`crate::streaming::stream_width`]); callers fall
/// back to a conservative default when this returns `None`, so a miss costs
/// concurrency tuning, never correctness. Never panics — same "return nothing
/// on any error" contract as [`query_iokit_gpu_utilization`].
pub fn gpu_core_count() -> Option<usize> {
    unsafe {
        let service_name = CString::new("IOAccelerator").ok()?;
        let matching = IOServiceMatching(service_name.as_ptr());
        if matching.is_null() {
            return None;
        }

        let mut iterator: io_iterator_t = 0;
        let ret = IOServiceGetMatchingServices(
            kIOMasterPortDefault,
            matching as CFDictionaryRef,
            &mut iterator,
        );
        if ret != 0 {
            return None;
        }

        let mut cores: Option<usize> = None;
        loop {
            let service = IOIteratorNext(iterator);
            if service == 0 {
                break;
            }
            let mut props: CFMutableDictionaryRef = std::ptr::null_mut();
            let pret = IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0);
            IOObjectRelease(service);
            if pret != 0 || props.is_null() {
                continue;
            }
            if let Some(v) = read_int_property(props as CFDictionaryRef, "gpu-core-count") {
                if v > 0 {
                    cores = Some(cores.map_or(v as usize, |c| c.max(v as usize)));
                }
            }
            CFRelease(props as CFTypeRef);
        }
        IOObjectRelease(iterator);
        cores
    }
}

/// Read a top-level signed-integer property from a service property dict,
/// or `None` if the key is absent / not a `CFNumber`.
unsafe fn read_int_property(props: CFDictionaryRef, key: &str) -> Option<i64> {
    let k = cfstr(key)?;
    let val = CFDictionaryGetValue(props, k as *const c_void);
    CFRelease(k as CFTypeRef);
    if val.is_null() || CFGetTypeID(val) != CFNumberGetTypeID() {
        return None;
    }
    let mut out: i64 = 0;
    let ok = CFNumberGetValue(
        val as CFNumberRef,
        kCFNumberSInt64Type,
        &mut out as *mut i64 as *mut c_void,
    );
    ok.then_some(out)
}

/// Read `PerformanceStatistics -> "Device Utilization %"` from one service's
/// property dictionary, or `None` if either key is absent / not a number.
unsafe fn read_device_utilization(props: CFDictionaryRef) -> Option<i64> {
    let perf_key = cfstr("PerformanceStatistics")?;
    let perf_dict = CFDictionaryGetValue(props, perf_key as *const c_void);
    CFRelease(perf_key as CFTypeRef);
    // Guard the borrowed value's dynamic type before treating it as a
    // dictionary: a driver populating an unexpected CF type would otherwise
    // make the CFDictionaryGetValue call below undefined behavior.
    if perf_dict.is_null() || CFGetTypeID(perf_dict) != CFDictionaryGetTypeID() {
        return None;
    }

    let util_key = cfstr("Device Utilization %")?;
    let util_val = CFDictionaryGetValue(perf_dict as CFDictionaryRef, util_key as *const c_void);
    CFRelease(util_key as CFTypeRef);
    if util_val.is_null() || CFGetTypeID(util_val) != CFNumberGetTypeID() {
        return None;
    }

    let mut val: i64 = 0;
    let ok = CFNumberGetValue(
        util_val as CFNumberRef,
        kCFNumberSInt64Type,
        &mut val as *mut i64 as *mut c_void,
    );
    ok.then_some(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live sensor read must never panic and must report a valid percentage.
    /// Exercises the real IOKit path (this crate is macOS-only).
    #[test]
    fn query_iokit_gpu_utilization_is_in_range() {
        let util = query_iokit_gpu_utilization();
        assert!(util <= 100, "util {util} out of 0..=100 range");
    }

    #[test]
    fn governor_without_yielding_never_throttles() {
        let mut gov = UtilGovernor::start(0, 100, false);
        assert!(!gov.should_throttle());
        assert_eq!(gov.utilization(), 0.0);
        gov.stop();
    }
}
