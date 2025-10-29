"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

from typing import Dict, List, Tuple, Any, Union, Mapping, Sequence, cast, Optional
import collections.abc
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler
import dimod
import dwave_networkx as dnx

from dwave_topologies.embedding_loader import get_embedding_dict, embedding_exists

# Type definitions to match base_miner
Variable = collections.abc.Hashable


class DWaveSamplerWrapper:
    """Wrapper class for D-Wave sampler with configuration management."""

    def __init__(self, topology_name: Optional[str] = None, job_label_prefix: Optional[str] = None):
        """
        Initialize D-Wave sampler wrapper.

        Args:
            topology_name: Optional topology name like "Z(10,2)" to use precomputed embedding.
                          If None, uses full QPU topology (default).
            job_label_prefix: Optional prefix for job labels on D-Wave dashboard (e.g., "QUIP_MINE").
                             Full label will be "{prefix}_{topology}" (e.g., "QUIP_MINE_10_2").
        """
        self.topology_name = topology_name
        self.job_label_prefix = job_label_prefix or "QUIP_MINE"

        # Initialize base QPU sampler
        base_sampler = DWaveSampler()
        self.qpu_solver = base_sampler

        # Determine topology and embedding
        if topology_name:
            # Use precomputed embedding for specific topology
            solver_name = base_sampler.properties.get('chip_id', 'Advantage2_system1.6')

            # Normalize solver name for directory (remove special chars)
            solver_dir = solver_name.replace('-', '_').replace('.', '_')

            # Check if embedding exists
            if not embedding_exists(topology_name, solver_dir):
                raise FileNotFoundError(
                    f"No precomputed embedding found for {topology_name} on {solver_name}. "
                    f"Run: python tools/analyze_topology_sizes.py --configs '{topology_name.strip('Z()').replace(',', ' ')}' --precompute-embedding"
                )

            # Load embedding
            embedding = get_embedding_dict(topology_name, solver_dir, convert_keys_to_int=True)
            if embedding is None:
                raise ValueError(f"Failed to load embedding for {topology_name}")

            # Create FixedEmbeddingComposite
            self.sampler = FixedEmbeddingComposite(base_sampler, embedding)
            self.embedding = embedding

            # Generate topology graph to get nodes/edges
            if topology_name.startswith("Z(") and topology_name.endswith(")"):
                parts = topology_name[2:-1].split(",")
                m, t = int(parts[0].strip()), int(parts[1].strip())
                topology_graph = dnx.zephyr_graph(m, t)
            else:
                raise ValueError(f"Unsupported topology format: {topology_name}")

            self.nodelist: List[Variable] = list(topology_graph.nodes())
            self.edgelist: List[Tuple[Variable, Variable]] = list(topology_graph.edges())

            # Build job label: QUIP_MINE_10_2
            topology_label = topology_name.strip("Z()").replace(",", "_").replace(" ", "")
            self.job_label = f"{self.job_label_prefix}_{topology_label}"

        else:
            # Use full QPU topology (no embedding)
            self.sampler = base_sampler
            self.embedding = None
            self.nodelist: List[Variable] = list(base_sampler.nodelist)
            self.edgelist: List[Tuple[Variable, Variable]] = list(base_sampler.edgelist)
            self.job_label = self.job_label_prefix

        self.is_qpu = True
        self.sampler_type = "qpu"
        self.properties: Dict[str, Any] = dict(base_sampler.properties)

        # For quantum_proof_of_work functions, nodes and edges should be int lists
        self.nodes: List[int] = cast(List[int], self.nodelist)
        self.edges: List[Tuple[int, int]] = cast(List[Tuple[int, int]], self.edgelist)

    def sample_ising(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample from the D-Wave QPU with automatic job labeling.

        Automatically adds 'label' parameter for D-Wave dashboard visibility.
        """
        # Add job label if not already specified
        if 'label' not in kwargs:
            kwargs['label'] = self.job_label

        return self.sampler.sample_ising(h, J, **kwargs)