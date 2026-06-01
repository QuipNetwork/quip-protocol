# Metal adaptive utilization governor

Keeps an interactive Mac responsive while mining on Apple Silicon, and recovers
full throughput when nobody is using the machine. Metal-only and **independent
of the CUDA util monitor** — it lives in `GPU/metal_scheduler.py` and
`GPU/macos_sensors.py` and shares no utilization machinery with
`GPU/util_monitor.py` (the only shared code is `shared/proc_util.py`'s generic
process lifecycle).

## The jank lever: GPU occupancy

Measured on a 40-core M4 Max with this SA kernel, UI smoothness is governed by
**concurrent GPU threads per command buffer = problems × reads** (occupancy) —
*not* core count, duty-cycle percentage, or command-buffer duration:

| problems × reads | threads | result |
|---|---:|---|
| 40 × 32 | 1,280 | smooth |
| 8 × 256 | 2,048 | smooth |
| 8 × 512 | 4,096 | stutters |
| 8 × 1024 | 8,192 | stutters |

Below ~2–3k concurrent threads the execution units / memory bandwidth aren't
saturated, so the compositor's work interleaves → smooth. Above ~4k it
saturates → the UI stalls. This threshold is hardware- and kernel-specific and
**not documented by Apple** (occupancy must be profiled); it's exposed as a
tunable with a measured default.

## How it works

When `yielding` is on (default), `MetalScheduler` spawns a Metal-specific
monitor process (`cap_monitor_main`) that polls the macOS sensors every ~0.3s,
runs a tier state machine with hysteresis, and publishes an **occupancy budget**
(threads/command buffer) plus a heartbeat:

| tier | condition | budget |
|---|---|---|
| PAUSE | on battery **or** thermal critical | `0` (stop) |
| LOW | thermal serious | `active_threads // 2` |
| IDLE | headless **or** idle > `idle_after_s` | `UNCAPPED` (full speed) |
| ACTIVE | otherwise (user present) | `active_threads` |

Hysteresis is asymmetric: enter a protective tier (PAUSE/LOW) immediately, leave
it only after `LEAVE_POLLS` clear polls; IDLE↔ACTIVE is immediate. If the
monitor heartbeat goes stale, `get_thread_budget` fails safe to `active_threads`
(never uncapped).

Each batch, `MetalSASampler.sample_ising_streaming` re-reads the budget and:

- `UNCAPPED` → one monolithic dispatch (full batch, full reads — max throughput),
- `0` → pause: idle and re-check (stop-aware, never blocks teardown),
- else → **read-split**: keep the full problem batch and full sweeps, but run
  the reads in command buffers of `budget // problems` each (distinct seed per
  buffer, concatenated) so `problems × reads ≤ budget`. Total reads preserved.

### Why read-splitting is safe

Each read is an independent SA run that executes the full sweep schedule on its
own — reads never interact. So running `num_reads` as several smaller buffers
with distinct seeds is equivalent to one big dispatch (the sweep parameter is
per-read and untouched). Splitting trades parallelism for a lower per-buffer
occupancy; this kernel is memory-bandwidth-bound, so the throughput cost of
staying under the smooth budget is modest.

## CPU-side politeness

On first entry into the Metal sampling path (in the stream-driver child), the
sampler lowers its thread QoS to `QOS_CLASS_UTILITY` (0x11) via `ctypes`, so it
yields P-cores to the foreground UI. It does not bound GPU occupancy — it
composes with the read-split cap. It is never applied from the shared QPU
stream driver.

## Config (`[metal]` section)

| key | default | meaning |
|---|---|---|
| `yielding` | `true` | master enable for the adaptive cap |
| `active_threads` | `2048` | occupancy budget (max concurrent GPU threads per command buffer) while present; the jank lever. Lower (e.g. 1024) if the UI still stutters; raise for more throughput while present |
| `idle_after_s` | `60` | seconds of no HID input before going IDLE (away) |
| `utilization` | `100` | problem-batch size (core budget = cores × util%) — unchanged role |

`active_threads`/`idle_after_s` are Metal-only and live in the `[metal]` device
section; they do not inherit from the shared `[gpu]` defaults (kept out of
`_GPU_CFG_KEYS`, so CUDA is unaffected). Idle/headless runs uncapped; battery
and critical thermal pause; thermal-serious halves the budget.

Sensors degrade to polite defaults on any failure (assume present, not battery,
nominal thermal) and never raise into the monitor loop.
