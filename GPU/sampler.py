"""Local GPU sampler using PyTorch (CUDA or MPS) in a persistent worker process."""

import os
import multiprocessing
from typing import Any, Dict, List, Tuple
from queue import Empty as QueueEmpty
import collections.abc

import dimod
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import create_topology_graph, get_topology_properties
from .worker import gpu_worker_main

Variable = collections.abc.Hashable

# Optional torch import
try:
    import torch
except ImportError:
    torch = None


class GPUSampler(MockDWaveSampler):
    """Local GPU sampler using PyTorch (CUDA or MPS) in a persistent worker process."""

    def __init__(self, device: str):
        self._device = str(device)
        self._debug = os.getenv("QUIP_DEBUG") == "1"
        self._ctx = multiprocessing.get_context("spawn")
        self._req_q: multiprocessing.Queue = self._ctx.Queue()
        self._resp_q: multiprocessing.Queue = self._ctx.Queue()
        self.sampler_type = "gpu"

        self._proc = self._ctx.Process(target=gpu_worker_main, args=(self._req_q, self._resp_q, self._device))
        self._proc.daemon = True
        self._proc.start()
        if self._debug:
            print(f"[GPU parent pid={os.getpid()}] spawn worker pid={self._proc.pid} device={self._device}", flush=True)

        # Use the default topology (Pegasus) from quantum_proof_of_work
        topology_graph = create_topology_graph()  # Uses DEFAULT_TOPOLOGY (Pegasus)
        properties = get_topology_properties()

        super().__init__(
            nodelist=list(topology_graph.nodes()),
            edgelist=list(topology_graph.edges()),
            properties=properties,
            substitute_sampler=self
        )

        self.nodelist: List[Variable] = list(topology_graph.nodes())
        self.edgelist: List[Tuple[Variable, Variable]] = list(topology_graph.edges())
        self.properties: Dict[str, Any] = properties
        
        # Type conversions to match protocol expectations (nodes should be ints for quantum_proof_of_work functions)
        nodes = []
        for node in self.nodelist:
            if not isinstance(node, int):
                raise ValueError(f"Expected node index to be int, got {type(node)}")
            nodes.append(int(node))
        edges = []
        for edge in self.edgelist:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError(f"Expected edge to be tuple of length 2, got {edge}")
            if not isinstance(edge[0], int) or not isinstance(edge[1], int):
                raise ValueError(f"Expected edge indices to be int, got {type(edge[0])} and {type(edge[1])}")
            edges.append((int(edge[0]), int(edge[1])))
        self.nodes = nodes
        self.edges = edges

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

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs) -> dimod.SampleSet:
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
        timeout = float(os.getenv("QUIP_GPU_WORKER_RESP_TIMEOUT", "30.0"))
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

        # Convert samples to the format expected by dimod.SampleSet.from_samples
        # samples should be a list of dicts mapping variables to values
        sample_dicts = []
        for sample in samples:
            sample_dict = {i: sample[i] for i in range(len(sample))}
            sample_dicts.append(sample_dict)
        
        # Create proper dimod.SampleSet
        return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)

