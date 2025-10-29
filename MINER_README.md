# Quantum Blockchain Miner Architecture

## 2-Process Mining Architecture

### Process Structure
```
Parent Process (Controller)
├── Monitors stop_event every 100ms
├── Manages child worker lifecycle  
└── Handles termination via SIGTERM/SIGKILL

Child Process (Mining Worker)
├── Runs existing mine_block() logic
├── Handles SIGTERM for resource cleanup
└── Communicates results via IPC
```

### Implementation

The parent process logic is implemented in `shared/miner_worker.py` via `MinerHandle.mine_with_timeout()`. Miners only need to:

1. **Implement SIGTERM handler** for hardware resource cleanup
2. **Use existing mine_block() logic** unchanged
3. **Register signal handler** in miner constructor

### Signal Handler Requirements

Each miner must implement cleanup in their `__init__()`:

```python
import signal

def __init__(self, miner_id: str, **cfg):
    # ... existing initialization ...
    signal.signal(signal.SIGTERM, self._cleanup_handler)

def _cleanup_handler(self, signum, frame):
    # Hardware-specific cleanup
    # CUDA: cudaDeviceReset()
    # QPU: Cancel D-Wave jobs
    # Metal: Release MPS resources
    # Modal: Terminate cloud functions
    exit(0)
```

### Process Flow

#### Normal Operation
1. `MinerHandle.mine_with_timeout()` spawns child process
2. Child runs existing `mine_block()` method
3. Parent monitors `stop_event` every 100ms
4. Child completes and returns result

#### Signal Interruption  
1. Parent detects `stop_event.is_set()`
2. Parent sends SIGTERM to child
3. Child's signal handler performs cleanup and exits
4. Parent waits 2s, then SIGKILL if needed
5. Parent returns None

### Testing Requirements

- Signal response within 200ms
- No resource leaks after termination
- Hardware remains functional after forced termination  
- Mining results identical to single-process version

### Hardware-Specific Cleanup

- **CPU**: Reset library state, deallocate memory
- **CUDA**: `cudaDeviceReset()`, clear streams/contexts
- **Metal**: Release MPS command buffers and memory
- **QPU**: Cancel D-Wave jobs via API, close connections
- **Modal**: Terminate cloud functions, cleanup sessions