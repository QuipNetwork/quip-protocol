"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

from typing import Dict, List, Tuple, Any, Union, Mapping, Sequence, cast, Optional, TYPE_CHECKING
import collections.abc
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler
from dwave.embedding import embed_bqm, unembed_sampleset
import dimod
import dwave_networkx as dnx

if TYPE_CHECKING:
    from dwave.cloud.computation import Future


class EmbeddedFuture:
    """Wrapper around a D-Wave Future that handles unembedding when sampleset is accessed.

    This enables async submission of embedded problems while still getting properly
    unembedded results when the future completes.
    """

    def __init__(self, future: 'Future', source_bqm: dimod.BinaryQuadraticModel,
                 embedding: Dict[int, List[int]], chain_strength: Optional[float] = None):
        """
        Args:
            future: The raw Future from the QPU sampler
            source_bqm: The original (unembedded) BQM for variable reference
            embedding: The embedding mapping {source_var: [target_qubits]}
            chain_strength: Chain strength used (for broken chain handling)
        """
        self._future = future
        self._source_bqm = source_bqm
        self._embedding = embedding
        self._chain_strength = chain_strength
        self._cached_sampleset: Optional[dimod.SampleSet] = None

    @property
    def sampleset(self) -> dimod.SampleSet:
        """Get the unembedded sampleset (blocks if not ready)."""
        if self._cached_sampleset is None:
            # Get raw embedded sampleset from QPU
            embedded_sampleset = self._future.sampleset

            # Unembed to get logical variable samples
            self._cached_sampleset = unembed_sampleset(
                embedded_sampleset,
                self._embedding,
                self._source_bqm,
                chain_break_method='majority_vote'
            )
        return self._cached_sampleset

    def done(self) -> bool:
        """Check if the future is complete."""
        return self._future.done()

    def cancel(self) -> bool:
        """Cancel the pending job."""
        return self._future.cancel()

    def wait(self, timeout: Optional[float] = None):
        """Wait for the future to complete."""
        return self._future.wait(timeout)

    @property
    def id(self):
        """Get the job ID."""
        return self._future.id

    def __hash__(self):
        """Make EmbeddedFuture hashable using the underlying future's id."""
        return hash(id(self._future))

    def __eq__(self, other):
        """Compare by underlying future identity."""
        if isinstance(other, EmbeddedFuture):
            return self._future is other._future
        return False

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
        solver_name = base_sampler.properties.get('chip_id', 'Advantage2_system1.10')
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
            topology_name: Name of the topology (e.g., "Z(9,2)" or "Advantage2_system1.10")
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

    def sample_ising_async(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> Union['Future', EmbeddedFuture]:
        """
        Submit Ising problem to QPU and return Future without blocking.

        Same as sample_ising() but returns a Future-like object for async processing.
        Caller must access future.sampleset to get results (which blocks on first access).

        For embedded problems, returns an EmbeddedFuture that handles unembedding
        automatically when the sampleset is accessed.

        Args:
            h: Linear biases (dict mapping variable to bias, or sequence)
            J: Quadratic biases (dict mapping variable pairs to bias)
            **kwargs: Additional parameters passed to the sampler (num_reads, annealing_time, etc.)

        Returns:
            Future-like object (Future or EmbeddedFuture) that resolves to SampleSet
        """
        # Add default job label if not already specified
        if 'label' not in kwargs:
            kwargs['label'] = self.job_label

        if self.embedding is not None:
            # Create BQM from Ising problem
            source_bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

            # Calculate chain strength (using same logic as FixedEmbeddingComposite)
            # Default to magnitude of strongest interaction
            if source_bqm.num_interactions > 0:
                chain_strength = max(abs(bias) for bias in source_bqm.quadratic.values()) * 1.5
            else:
                chain_strength = max(abs(bias) for bias in source_bqm.linear.values()) * 1.5 if source_bqm.linear else 1.0

            # Manually embed the BQM
            target_bqm = embed_bqm(
                source_bqm,
                self.embedding,
                self.qpu_solver.adjacency,
                chain_strength=chain_strength
            )

            # Submit embedded BQM directly to QPU's underlying solver (returns raw Future)
            # DWaveSampler.sample() returns SampleSet, but solver.sample_bqm() returns Future
            raw_future = self.qpu_solver.solver.sample_bqm(target_bqm, **kwargs)

            # Wrap in EmbeddedFuture to handle unembedding on access
            return EmbeddedFuture(
                future=raw_future,
                source_bqm=source_bqm,
                embedding=self.embedding,
                chain_strength=chain_strength
            )
        else:
            # No embedding - submit to underlying solver directly (returns raw Future)
            bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
            return self.qpu_solver.solver.sample_bqm(bqm, **kwargs)