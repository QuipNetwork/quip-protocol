"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

from typing import Dict, List, Tuple, Any, Union, Mapping, Sequence, cast, Optional
import collections.abc
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler
import dimod
import dwave_networkx as dnx

from dwave_topologies.embedding_loader import get_embedding_dict, embedding_exists
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology

# Type definitions to match base_miner
Variable = collections.abc.Hashable


class DWaveSamplerWrapper:
    """Wrapper class for D-Wave sampler with configuration management.

    This sampler encapsulates embedding logic internally. Callers always work with
    logical topology variables, and the sampler handles mapping to physical qubits.
    """

    def __init__(
        self,
        topology: DWaveTopology = DEFAULT_TOPOLOGY,
        embedding_file: Optional[str] = None,
        job_label_prefix: Optional[str] = None
    ):
        """
        Initialize D-Wave sampler wrapper.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY = Z(9,2)).
                     Can be any DWaveTopology (Zephyr, Advantage2, etc.)
            embedding_file: Optional path to embedding file. If None and topology requires
                          embedding, will search for precomputed embedding.
            job_label_prefix: Optional prefix for job labels on D-Wave dashboard.
                             If None, generates format like "Quip_Z9_T2" for Zephyr,
                             "Quip_C16" for Chimera, "Quip_P16" for Pegasus.
        """
        self.topology = topology
        self.topology_name = topology.solver_name

        # Generate default job label prefix based on topology type
        if job_label_prefix is None:
            # Extract topology type and parameters
            if hasattr(topology, 'm') and hasattr(topology, 't'):
                # Zephyr topology
                job_label_prefix = f"Quip_Z{topology.m}_T{topology.t}"
            elif hasattr(topology, 'M'):
                # Chimera topology (C_M)
                job_label_prefix = f"Quip_C{topology.M}"
            elif hasattr(topology, 'P'):
                # Pegasus topology (P_P)
                job_label_prefix = f"Quip_P{topology.P}"
            else:
                # Generic/hardware topology - use solver name
                job_label_prefix = f"Quip_{topology.solver_name.replace('.', '_').replace('-', '_')}"

        self.job_label_prefix = job_label_prefix

        # Initialize base QPU sampler
        base_sampler = DWaveSampler()
        self.qpu_solver = base_sampler

        # Get hardware info
        solver_name = base_sampler.properties.get('chip_id', 'Advantage2_system1.7')
        solver_dir = solver_name.replace('-', '_').replace('.', '_')

        # Determine if this topology needs embedding
        needs_embedding = self._needs_embedding(topology.solver_name, solver_name)

        if needs_embedding:
            # Load embedding (either specified or auto-discover)
            if embedding_file:
                # Load specified embedding file
                import gzip
                import json
                with gzip.open(embedding_file, 'rt') as f:
                    embedding_data = json.load(f)
                    embedding = {int(k): v for k, v in embedding_data.items()}
            else:
                # Auto-discover precomputed embedding
                if not embedding_exists(topology.solver_name, solver_dir):
                    # Try to provide helpful error message
                    if topology.solver_name.startswith("Z("):
                        config = topology.solver_name.strip('Z()').replace(',', ' ')
                        hint = f"  python tools/analyze_topology_sizes.py --configs '{config}' --precompute-embedding"
                    else:
                        hint = f"  (No auto-generation available for {topology.solver_name})"

                    raise FileNotFoundError(
                        f"No precomputed embedding found for {topology.solver_name} on {solver_name}. "
                        f"Either provide embedding_file parameter or precompute embedding with:\n{hint}"
                    )

                embedding = get_embedding_dict(topology.solver_name, solver_dir, convert_keys_to_int=True)
                if embedding is None:
                    raise ValueError(f"Failed to load embedding for {topology.solver_name}")

            # Create FixedEmbeddingComposite (encapsulated internally)
            self.sampler = FixedEmbeddingComposite(base_sampler, embedding)
            self.embedding = embedding

            # Use topology's graph directly
            self.nodelist: List[Variable] = topology.nodes
            self.edgelist: List[Tuple[Variable, Variable]] = topology.edges

        else:
            # Native hardware topology - no embedding needed
            self.sampler = base_sampler
            self.embedding = None
            self.nodelist: List[Variable] = topology.nodes
            self.edgelist: List[Tuple[Variable, Variable]] = topology.edges

        # Job label is just the prefix (which already contains topology info)
        self.job_label = self.job_label_prefix

        self.is_qpu = True
        self.sampler_type = "qpu"
        self.properties: Dict[str, Any] = dict(base_sampler.properties)

        # For quantum_proof_of_work functions, nodes and edges should be int lists
        self.nodes: List[int] = cast(List[int], self.nodelist)
        self.edges: List[Tuple[int, int]] = cast(List[Tuple[int, int]], self.edgelist)

    def _needs_embedding(self, topology_name: str, solver_name: str) -> bool:
        """
        Determine if a topology needs embedding to run on the QPU.

        Args:
            topology_name: Name of the topology (e.g., "Z(9,2)" or "Advantage2_system1.7")
            solver_name: Name of the QPU solver

        Returns:
            True if embedding is needed, False if native topology
        """
        # Native hardware topologies don't need embedding
        solver_normalized = solver_name.replace('-', '_').replace('.', '_')
        topology_normalized = topology_name.replace('-', '_').replace('.', '_')

        if topology_normalized == solver_normalized:
            return False

        # Zephyr topologies need embedding (support both old and new formats)
        # New format: "Z(9,2)"
        # Old format (deprecated): "Zephyr_Z9_T2_Generic"
        if topology_name.startswith("Z(") or topology_name.startswith("Zephyr_Z"):
            return True

        # Unknown topology format
        raise ValueError(
            f"Cannot determine if topology '{topology_name}' needs embedding. "
            f"Expected Zephyr format 'Z(m,t)' or native hardware name matching solver '{solver_name}'"
        )

    def sample_ising(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample from the D-Wave QPU with automatic job labeling.

        Automatically adds 'label' parameter for D-Wave dashboard visibility.
        Caller can override by passing 'label' in kwargs.

        For embedded samplers, converts Ising to BQM explicitly to ensure
        correct variable labeling during unembedding.
        """
        # Add default job label if not already specified
        if 'label' not in kwargs:
            kwargs['label'] = self.job_label

        # For FixedEmbeddingComposite, we need to be explicit about variable labels
        # to ensure proper unembedding. Create a BQM from h, J with explicit labels.
        if self.embedding is not None:
            # Create BQM with explicit integer variable labels matching embedding keys
            bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

            # Verify BQM variables match embedding keys
            bqm_vars = set(bqm.variables)
            embedding_vars = set(self.embedding.keys())

            if bqm_vars != embedding_vars:
                import sys
                print(f"\n⚠️  WARNING: BQM variables don't match embedding keys!", file=sys.stderr)
                print(f"   BQM vars: {len(bqm_vars)}, range: {min(bqm_vars)}-{max(bqm_vars)}", file=sys.stderr)
                print(f"   Embedding vars: {len(embedding_vars)}, range: {min(embedding_vars)}-{max(embedding_vars)}", file=sys.stderr)

            # Sample using BQM (not sample_ising)
            sampleset = self.sampler.sample(bqm, **kwargs)
        else:
            # No embedding, use sample_ising directly
            sampleset = self.sampler.sample_ising(h, J, **kwargs)

        # Verify the variables match the expected logical topology
        if self.embedding is not None:
            expected_vars = set(self.nodelist)
            actual_vars = set(sampleset.variables)

            if actual_vars != expected_vars:
                import sys
                print(f"\n⚠️  WARNING: Sampleset variables don't match logical topology!", file=sys.stderr)
                print(f"   Expected: {len(expected_vars)} vars (0-{max(expected_vars)})", file=sys.stderr)
                print(f"   Got: {len(actual_vars)} vars ({min(actual_vars)}-{max(actual_vars)})", file=sys.stderr)
                print(f"   Missing: {sorted(list(expected_vars - actual_vars))[:20]}", file=sys.stderr)
                print(f"   Extra: {sorted(list(actual_vars - expected_vars))[:20]}", file=sys.stderr)

        return sampleset