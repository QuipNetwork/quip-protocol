//! NVML utilization governor (port of `GPU/gpu_scheduler.py` yielding path).
//!
//! Static util ceiling from config; optional background poll when yielding.
//! When yielding and observed util > 90%, the session loop inserts a brief
//! pause so sibling GPU users get time slices.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Shared utilization sample (0–100) and governor knobs.
pub struct UtilGovernor {
    /// Config ceiling 1–100.
    pub utilization_ceiling: u32,
    pub yielding: bool,
    last_util: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl UtilGovernor {
    /// Start an NVML poller when `yielding` is true and NVML is available.
    ///
    /// Falls back to a silent no-op governor if NVML init fails (miner still
    /// runs; utilization stays 0 and throttle never fires).
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let ceiling = utilization_ceiling.clamp(1, 100);
        let last_util = Arc::new(AtomicU32::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = if yielding {
            let last_c = Arc::clone(&last_util);
            let stop_c = Arc::clone(&stop);
            Some(thread::spawn(move || {
                poll_loop(device_index, last_c, stop_c)
            }))
        } else {
            None
        };
        Self {
            utilization_ceiling: ceiling,
            yielding,
            last_util,
            stop,
            handle,
        }
    }

    /// Last NVML GPU util percent (0–100), or 0 if not yielding / unavailable.
    pub fn utilization(&self) -> f32 {
        self.last_util.load(Ordering::Relaxed) as f32
    }

    /// True when yielding and last NVML sample exceeds 90%.
    pub fn should_throttle(&self) -> bool {
        self.yielding && self.last_util.load(Ordering::Relaxed) > 90
    }

    /// Request the poller to exit and join it.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
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

fn poll_loop(device_index: u32, last: Arc<AtomicU32>, stop: Arc<AtomicBool>) {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return;
    };
    let Ok(device) = nvml.device_by_index(device_index) else {
        return;
    };
    while !stop.load(Ordering::Relaxed) {
        if let Ok(rates) = device.utilization_rates() {
            last.store(rates.gpu, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_secs(2));
    }
}
