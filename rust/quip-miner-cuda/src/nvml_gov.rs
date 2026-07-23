//! NVML utilization governor (port of `GPU/gpu_scheduler.py` yielding path).
//!
//! Util ceiling + yielding are runtime knobs (atomics): the CLI sets them at
//! launch, and [`UtilGovernor::reconfigure`] overrides them when the
//! coordinator's `Configure` arrives. When yielding and the observed GPU util
//! exceeds the ceiling, the session loop inserts a brief pause so sibling GPU
//! users get time slices.

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
    /// Last NVML GPU util percent 0–100 (0 while not yielding).
    last_util: AtomicU32,
    stop: AtomicBool,
}

/// Shared utilization sample and reconfigurable governor knobs.
pub struct UtilGovernor {
    knobs: Arc<Knobs>,
    handle: Option<JoinHandle<()>>,
}

impl UtilGovernor {
    /// Start the NVML poller. Values come from the CLI; `Configure` may later
    /// override them via [`reconfigure`](Self::reconfigure). The poll thread
    /// runs regardless of `yielding` (so a later `false -> true` override starts
    /// sampling with no thread churn) but only records util while yielding.
    ///
    /// Falls back to a silent no-op if NVML init fails (miner still runs; util
    /// stays 0 and throttle never fires).
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let knobs_thread = Arc::clone(&knobs);
        let handle = Some(thread::spawn(move || poll_loop(device_index, &knobs_thread)));
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

    /// Last NVML GPU util percent (0–100), or 0 if not yielding / unavailable.
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

fn poll_loop(device_index: u32, knobs: &Knobs) {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return;
    };
    let Ok(device) = nvml.device_by_index(device_index) else {
        return;
    };
    while !knobs.stop.load(Ordering::Relaxed) {
        if knobs.yielding.load(Ordering::Relaxed) {
            if let Ok(rates) = device.utilization_rates() {
                knobs.last_util.store(rates.gpu, Ordering::Relaxed);
            }
        } else {
            knobs.last_util.store(0, Ordering::Relaxed);
        }
        thread::sleep(Duration::from_secs(2));
    }
}
