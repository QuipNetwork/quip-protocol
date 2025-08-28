"""Local GPU sampler using PyTorch (CUDA or MPS) in a persistent worker process."""

import os
import multiprocessing
import numpy as np
from queue import Empty as QueueEmpty

from dwave.system.testing import MockDWaveSampler
from .worker import gpu_worker_main

# Optional torch import
try:
    import torch
except ImportError:
    torch = None


class LocalGPUSampler(MockDWaveSampler):
    """Local GPU sampler using PyTorch (CUDA or MPS) in a persistent worker process."""

    def __init__(self, device: str):
        self._device = str(device)
        self._debug = os.getenv("QUIP_DEBUG") == "1"
        self._ctx = multiprocessing.get_context("spawn")
        self._req_q: multiprocessing.Queue = self._ctx.Queue()
        self._resp_q: multiprocessing.Queue = self._ctx.Queue()
        
        self._proc = self._ctx.Process(target=gpu_worker_main, args=(self._req_q, self._resp_q, self._device))
        self._proc.daemon = True
        self._proc.start()
        if self._debug:
            print(f"[GPU parent pid={os.getpid()}] spawn worker pid={self._proc.pid} device={self._device}", flush=True)

        # Use same topology as SimulatedAnnealingStructuredSampler
        qpu = MockDWaveSampler()
        super().__init__(
            nodelist=qpu.nodelist,
            edgelist=qpu.edgelist,
            properties=qpu.properties,
            substitute_sampler=self
        )

    def close(self):
        try:
            self._req_q.put({"op": "stop"})
        except Exception:
            pass
        try:
            if self._proc.is_alive():
                self._proc.join(timeout=2)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs):
        # Convert to dicts for serialization
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        payload = {
            "op": "sample",
            "h": h_dict,
            "J": J_dict,
            "num_reads": int(num_reads),
            "num_sweeps": int(num_sweeps),
        }
        if self._debug:
            print(f"[GPU parent pid={os.getpid()}] send sample to worker pid={self._proc.pid} device={self._device} reads={payload['num_reads']} sweeps={payload['num_sweeps']}", flush=True)
        self._req_q.put(payload)
        timeout = float(os.getenv("QUIP_GPU_WORKER_RESP_TIMEOUT", "5.0"))
        try:
            msg = self._resp_q.get(timeout=timeout)
        except QueueEmpty:
            raise RuntimeError(f"GPU worker timeout after {timeout}s (pid={self._proc.pid}, device={self._device})")
        if isinstance(msg, dict) and msg.get("status") == "error":
            raise RuntimeError(msg.get("message", "GPU worker error"))
        if self._debug:
            print(f"[GPU parent pid={os.getpid()}] received response from worker pid={self._proc.pid} device={self._device}", flush=True)
        samples = msg["samples"]
        energies = msg["energies"]

        class SampleSet:
            def __init__(self, samples, energies):
                self.record = type('Record', (), {
                    'sample': np.array(samples),
                    'energy': np.array(energies)
                })()
        return SampleSet(samples, energies)