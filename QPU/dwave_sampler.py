"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

import logging
import os
from typing import Dict, List, Tuple, Any, Union, Mapping, Sequence, cast, Optional, TYPE_CHECKING
import collections.abc
import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite

logger = logging.getLogger(__name__)
from dwave.embedding import embed_bqm, unembed_sampleset
import dimod

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

class ClampedFuture:
    """Wrapper that reconstructs full-topology solutions from a reduced QPU future.

    When defective qubits are clamped, the QPU solves a reduced problem.
    This future inserts fixed spins and recomputes energies on sampleset access.
    """

    def __init__(
        self,
        inner_future,
        fixed_spins: Dict[int, int],
        full_h: Dict[int, float],
        full_J: Dict[Tuple[int, int], float],
    ):
        self._inner = inner_future
        self._fixed_spins = fixed_spins
        self._full_h = full_h
        self._full_J = full_J
        self._cached_sampleset: Optional[dimod.SampleSet] = None

    @property
    def sampleset(self) -> dimod.SampleSet:
        """Get full-topology sampleset (blocks if not ready)."""
        if self._cached_sampleset is None:
            reduced = self._inner.sampleset
            full_bqm = dimod.BinaryQuadraticModel.from_ising(
                self._full_h, self._full_J
            )
            samples = []
            energies = []
            for sample in reduced.samples():
                full_sample = dict(sample)
                full_sample.update(self._fixed_spins)
                samples.append(full_sample)
                energies.append(full_bqm.energy(full_sample))

            info = dict(reduced.info) if hasattr(reduced, 'info') else {}
            self._cached_sampleset = dimod.SampleSet.from_samples(
                samples, vartype=dimod.SPIN, energy=energies, info=info,
            )
        return self._cached_sampleset

    def done(self) -> bool:
        return self._inner.done()

    def cancel(self) -> bool:
        return self._inner.cancel()

    def wait(self, timeout: Optional[float] = None):
        return self._inner.wait(timeout)

    @property
    def id(self):
        return self._inner.id

    def __hash__(self):
        return hash(id(self._inner))

    def __eq__(self, other):
        if isinstance(other, ClampedFuture):
            return self._inner is other._inner
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
        job_label_prefix: Optional[str] = None,
        solver_name: Optional[str] = None,
        region: Optional[str] = None,
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
            solver_name: Optional explicit solver name to connect to.
                        If None, uses DWAVE_API_SOLVER env var.
            region: Optional D-Wave region (e.g. "na-east-1").
                   If None, uses default from config.
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

        # Check for API key before attempting connection
        api_key = os.environ.get('DWAVE_API_KEY')
        if not api_key:
            logger.warning("[QPU] DWAVE_API_KEY environment variable not set!")
        else:
            logger.debug(f"[QPU] DWAVE_API_KEY set (length: {len(api_key)})")

        # Initialize base QPU sampler
        logger.info("[QPU] Connecting to D-Wave API...")
        try:
            sampler_kwargs: Dict[str, Any] = {'request_timeout': (60, 300)}
            if solver_name is not None:
                sampler_kwargs['solver'] = solver_name
            if region is not None:
                sampler_kwargs['region'] = region
            base_sampler = DWaveSampler(**sampler_kwargs)
            logger.info(f"[QPU] Connected to solver: {base_sampler.properties.get('chip_id', 'unknown')}")
            logger.info(f"[QPU] Qubits available: {len(base_sampler.nodelist)}")
        except Exception as e:
            logger.error(f"[QPU] Failed to connect to D-Wave: {e}")
            raise
        self.qpu_solver = base_sampler

        # Get hardware info
        hw_solver_name = base_sampler.properties.get('chip_id', 'Advantage2_system1')
        solver_dir = hw_solver_name.replace('-', '_').replace('.', '_')

        # Determine if this topology needs embedding
        try:
            needs_embedding = self._needs_embedding(topology.solver_name, hw_solver_name)
        except ValueError:
            # Topology doesn't match solver and isn't a known embeddable type.
            # This is normal when the stored topology is from a different
            # revision of the same hardware (e.g., System1.10 vs System1).
            # Keep the stored topology as the protocol reference and let
            # defect detection handle the qubit differences.
            logger.info(
                f"[QPU] Topology '{topology.solver_name}' doesn't match solver "
                f"'{hw_solver_name}' — using stored topology with defect detection"
            )
            needs_embedding = False

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
            self._defective_qubits: List[int] = []  # Embedding handles defects
            logger.info(f"[QPU] Embedding loaded: {len(embedding)} logical qubits mapped to hardware")

            # Use topology's graph directly
            self.nodelist: List[Variable] = topology.nodes
            self.edgelist: List[Tuple[Variable, Variable]] = topology.edges

        else:
            # Native hardware topology - no embedding needed
            logger.info(f"[QPU] Using native hardware topology (no embedding needed)")
            self.sampler = base_sampler
            self.embedding = None
            if topology is not None:
                # Detect defective qubits by comparing stored vs live topology
                live_node_set = set(base_sampler.nodelist)
                stored_node_set = set(topology.nodes)

                self._defective_qubits: List[int] = sorted(
                    stored_node_set - live_node_set
                )
                extra_qubits = sorted(live_node_set - stored_node_set)

                if self._defective_qubits:
                    logger.warning(
                        f"[QPU] {len(self._defective_qubits)} defective qubits "
                        f"(offline on live QPU): "
                        f"{self._defective_qubits[:20]}"
                        f"{'...' if len(self._defective_qubits) > 20 else ''}"
                        f" — will use variable clamping"
                    )
                else:
                    logger.info(
                        "[QPU] Live topology matches stored topology "
                        f"({len(stored_node_set)} qubits)"
                    )
                if extra_qubits:
                    logger.info(
                        f"[QPU] Live QPU has {len(extra_qubits)} extra "
                        f"qubits not in stored topology (ignored)"
                    )

                # ALWAYS use the full stored topology for nodes/edges
                # (consensus requires all miners solve the same problem)
                self.nodelist: List[Variable] = topology.nodes
                self.edgelist: List[Tuple[Variable, Variable]] = topology.edges
            else:
                # Use solver's own hardware graph (no stored topology)
                self._defective_qubits: List[int] = []
                self.nodelist: List[Variable] = sorted(base_sampler.nodelist)
                self.edgelist: List[Tuple[Variable, Variable]] = list(base_sampler.edgelist)

        # Job label is just the prefix (which already contains topology info)
        self.job_label = self.job_label_prefix

        self.is_qpu = True
        self.sampler_type = "qpu"
        self.properties: Dict[str, Any] = dict(base_sampler.properties)

        # For quantum_proof_of_work functions, nodes and edges should be int lists
        self.nodes: List[int] = cast(List[int], self.nodelist)
        self.edges: List[Tuple[int, int]] = cast(List[Tuple[int, int]], self.edgelist)

    def close(self):
        """Release QPU connection resources (Ocean SDK 9.x resource management)."""
        if hasattr(self, 'qpu_solver'):
            self.qpu_solver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _needs_embedding(self, topology_name: str, solver_name: str) -> bool:
        """
        Determine if a topology needs embedding to run on the QPU.

        Args:
            topology_name: Name of the topology (e.g., "Z(9,2)" or "Advantage2_system1")
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

    def _clamp_defective_qubits(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        nonce_seed: int,
    ) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], Dict[int, int]]:
        """Clamp defective qubits to deterministic spins and adjust neighbors.

        For each offline qubit k, assigns a fixed spin s_k (deterministic from
        nonce_seed) and absorbs its coupling energy into neighbors' h-fields:
            h'[j] += J[k,j] * s_k  for all neighbors j of k

        This preserves the energy contribution of the clamped qubit in the
        reduced problem, so the QPU optimizes the remaining variables correctly.

        Args:
            h: Linear biases for all nodes (full topology).
            J: Quadratic biases for all edges (full topology).
            nonce_seed: Seed for deterministic spin assignment (from block nonce).

        Returns:
            (h_reduced, J_reduced, fixed_spins) where:
            - h_reduced: biases without defective qubits (neighbors adjusted)
            - J_reduced: couplings without edges involving defective qubits
            - fixed_spins: {qubit_id: spin_value} for solution reconstruction
        """
        defective_set = set(self._defective_qubits)
        rng = np.random.default_rng(nonce_seed)

        # Assign deterministic ±1 spins to defective qubits
        fixed_spins: Dict[int, int] = {}
        for qubit in self._defective_qubits:
            fixed_spins[qubit] = int(2 * rng.integers(2) - 1)

        # Copy h, remove defective qubits, adjust neighbors
        h_reduced = {k: v for k, v in h.items() if k not in defective_set}

        for (u, v), j_val in J.items():
            if u in defective_set and v not in defective_set:
                # u is clamped, absorb into v's h-field
                h_reduced[v] = h_reduced.get(v, 0.0) + j_val * fixed_spins[u]
            elif v in defective_set and u not in defective_set:
                # v is clamped, absorb into u's h-field
                h_reduced[u] = h_reduced.get(u, 0.0) + j_val * fixed_spins[v]
            # If both are defective, energy is constant — skip

        # Remove all edges involving defective qubits
        J_reduced = {
            (u, v): val for (u, v), val in J.items()
            if u not in defective_set and v not in defective_set
        }

        return h_reduced, J_reduced, fixed_spins

    def _reconstruct_full_sampleset(
        self,
        reduced_sampleset: dimod.SampleSet,
        fixed_spins: Dict[int, int],
        full_h: Dict[int, float],
        full_J: Dict[Tuple[int, int], float],
    ) -> dimod.SampleSet:
        """Reconstruct full-topology sampleset from reduced QPU results.

        Inserts fixed spins back into each sample and recomputes energies
        on the full Ising model so validators see correct values.

        Args:
            reduced_sampleset: QPU results with defective qubits missing.
            fixed_spins: {qubit_id: spin_value} from clamping.
            full_h: Full topology linear biases.
            full_J: Full topology quadratic biases.

        Returns:
            SampleSet with all topology variables and recomputed energies.
        """
        full_bqm = dimod.BinaryQuadraticModel.from_ising(full_h, full_J)
        all_vars = sorted(full_bqm.variables)

        samples = []
        energies = []
        for sample in reduced_sampleset.samples():
            # Start with QPU results, add fixed spins
            full_sample = dict(sample)
            full_sample.update(fixed_spins)
            samples.append(full_sample)
            energies.append(full_bqm.energy(full_sample))

        # Preserve timing info from original sampleset
        info = dict(reduced_sampleset.info) if hasattr(reduced_sampleset, 'info') else {}

        return dimod.SampleSet.from_samples(
            samples,
            vartype=dimod.SPIN,
            energy=energies,
            info=info,
        )

    def sample_ising(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> dimod.SampleSet:
        """Sample from the D-Wave QPU with automatic job labeling.

        If defective qubits were detected at init, transparently clamps them
        to deterministic spins, submits the reduced problem, and reconstructs
        the full-topology sampleset before returning.

        Pass nonce_seed=<int> in kwargs for deterministic clamping (required
        when defective qubits are present).
        """
        # Pop clamping seed before passing to D-Wave
        nonce_seed = kwargs.pop('nonce_seed', None)

        # Handle defective qubits via variable clamping
        if self._defective_qubits and nonce_seed is not None:
            h_dict = dict(h) if not isinstance(h, dict) else h
            J_dict = dict(J) if not isinstance(J, dict) else J
            h_reduced, J_reduced, fixed_spins = self._clamp_defective_qubits(
                h_dict, J_dict, nonce_seed
            )
            sampleset = self._sample_ising_inner(h_reduced, J_reduced, **kwargs)
            return self._reconstruct_full_sampleset(
                sampleset, fixed_spins, h_dict, J_dict
            )

        return self._sample_ising_inner(h, J, **kwargs)

    def _sample_ising_inner(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> dimod.SampleSet:
        """Submit Ising problem to QPU (handles embedding transparently)."""
        # Add default job label if not already specified
        if 'label' not in kwargs:
            kwargs['label'] = self.job_label

        # Pop custom kwargs before passing to D-Wave
        chain_strength_multiplier = kwargs.pop('chain_strength_multiplier', 1.5)

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

            # Calculate chain strength explicitly so we control the multiplier
            if bqm.num_interactions > 0:
                chain_strength = max(abs(b) for b in bqm.quadratic.values()) * chain_strength_multiplier
            else:
                chain_strength = max(abs(b) for b in bqm.linear.values()) * chain_strength_multiplier if bqm.linear else 1.0

            # Sample using BQM (not sample_ising)
            sampleset = self.sampler.sample(bqm, chain_strength=chain_strength, **kwargs)
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
    ) -> Union['Future', EmbeddedFuture, ClampedFuture]:
        """Submit Ising problem to QPU and return Future without blocking.

        Same as sample_ising() but returns a Future-like object for async
        processing. Caller must access future.sampleset to get results
        (which blocks on first access).

        For defective qubits, wraps in ClampedFuture that reconstructs
        the full-topology solution on sampleset access.

        Args:
            h: Linear biases (dict mapping variable to bias, or sequence).
            J: Quadratic biases (dict mapping variable pairs to bias).
            **kwargs: Additional parameters (num_reads, annealing_time, etc.)
                Pass nonce_seed=<int> for deterministic clamping.

        Returns:
            Future-like object that resolves to full-topology SampleSet.
        """
        # Pop clamping seed before passing to D-Wave
        nonce_seed = kwargs.pop('nonce_seed', None)

        # Handle defective qubits via variable clamping
        if self._defective_qubits and nonce_seed is not None:
            h_dict = dict(h) if not isinstance(h, dict) else h
            J_dict = dict(J) if not isinstance(J, dict) else J
            h_reduced, J_reduced, fixed_spins = self._clamp_defective_qubits(
                h_dict, J_dict, nonce_seed
            )
            inner_future = self._sample_ising_async_inner(
                h_reduced, J_reduced, **kwargs
            )
            return ClampedFuture(
                inner_future=inner_future,
                fixed_spins=fixed_spins,
                full_h=h_dict,
                full_J=J_dict,
            )

        return self._sample_ising_async_inner(h, J, **kwargs)

    def _sample_ising_async_inner(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> Union['Future', EmbeddedFuture]:
        """Submit Ising problem to QPU async (handles embedding transparently)."""
        # Add default job label if not already specified
        if 'label' not in kwargs:
            kwargs['label'] = self.job_label

        # Pop custom kwargs before passing to D-Wave
        chain_strength_multiplier = kwargs.pop('chain_strength_multiplier', 1.5)

        if self.embedding is not None:
            # Create BQM from Ising problem
            source_bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

            # Calculate chain strength (using same logic as FixedEmbeddingComposite)
            # Default to magnitude of strongest interaction
            if source_bqm.num_interactions > 0:
                chain_strength = max(abs(bias) for bias in source_bqm.quadratic.values()) * chain_strength_multiplier
            else:
                chain_strength = max(abs(bias) for bias in source_bqm.linear.values()) * chain_strength_multiplier if source_bqm.linear else 1.0

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