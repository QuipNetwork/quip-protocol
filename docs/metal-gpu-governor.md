# Metal adaptive utilization governor

Keeps an interactive Mac responsive while mining on Apple Silicon, and recovers
full throughput when nobody is using the machine. Metal-only and **independent
of the CUDA util monitor** — it lives in `GPU/metal_scheduler.py` and
`GPU/macos_sensors.py` and shares no utilization machinery with
`GPU/util_monitor.py` (the only shared code is `shared/proc_util.py`'s generic
process lifecycle).

## How it works

When `yielding` is on (default), `MetalScheduler` spawns a Metal-specific
monitor process (`cap_monitor_main`) that polls the macOS sensors every ~0.3s
and publishes three shared ints the sampler reads lock-free:

- `target_pct` — the cap the sampler honors (0 = pause … 100 = flat-out)
- `measured_gpu` — IOKit GPU residency (a slow closed-loop trim signal)
- `heartbeat` — incremented each poll; staleness ⇒ fail-safe to `active_util`

Each batch, `MetalSASampler.sample_ising_streaming` re-reads `target_pct` and
picks its dispatch path:

| target_pct | path |
|---|---|
| 100 | monolithic `_dispatch_batch` (max throughput) |
| 0 | pause: idle and re-check (no models consumed) |
| 0 < t < 100 | adaptive chunked dispatch — burst-bounded + duty-cycled |

### Tier state machine (pure `decide_tier`)

Highest-priority satisfied row wins, with asymmetric hysteresis (enter a
protective tier immediately; leave it only after `LEAVE_POLLS` consecutive
clear polls):

| priority | condition | tier | cap |
|---|---|---|---|
| 1 | on battery **or** thermal critical | PAUSE | 0 |
| 2 | thermal serious | LOW | `serious_util` |
| 3 | headless **or** idle > `idle_after_s` | IDLE | `utilization` |
| 4 | otherwise (user present) | ACTIVE | `active_util` |

### The dual cap (any throttled tier)

1. **Burst bound** — `betas_per_chunk` is sized from a per-beta wall-time EMA so
   each committed command buffer runs ≈ `burst_ms` (~one frame), giving the
   compositor a preemption seam. This is the real jank lever; macOS only
   preempts the GPU at command-buffer boundaries.
2. **Average occupancy** — `DutyCycleController` paces sleeps between chunks to
   hold the average at `target_pct`, with a slow PI trim fed by `measured_gpu`.

A coordinated `DutyCycleController.set_target` resets the EMA + PI integral
together on a tier change, so no windup carries across caps.

### CPU-side politeness

On first entry into the Metal sampling path (in the stream-driver child), the
sampler lowers its thread QoS to `QOS_CLASS_UTILITY` (0x11) via `ctypes`, so it
yields P-cores to the foreground UI. It does not bound GPU occupancy — it
composes with the burst/duty cap. It is never applied from the shared QPU
stream driver.

## Config (`[metal]` section)

| key | default | meaning |
|---|---|---|
| `yielding` | `true` | master enable for the adaptive cap |
| `utilization` | `100` | IDLE/headless cap (flat-out when nobody's here) |
| `active_util` | `70` | cap while the user is present; drop to ~30 if janky |
| `idle_after_s` | `60` | seconds of no HID input before going flat-out |
| `burst_ms` | `8` | per-command-buffer wall-clock budget (≈ one frame) |
| `serious_util` | `30` | cap at thermal == serious |

These keys are Metal-only and live in the `[metal]` device section; they do not
inherit from the shared `[gpu]` defaults (kept out of `_GPU_CFG_KEYS` so CUDA is
unaffected). With shipped defaults a headless box stays flat-out (always IDLE);
an interactive box throttles to 70% while in use.

Sensors degrade to polite defaults on any failure (assume present, not battery,
nominal thermal) and never raise into the monitor loop.
