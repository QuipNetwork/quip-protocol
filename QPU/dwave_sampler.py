"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

import base64
import logging
import os
import time
from concurrent.futures import (
    FIRST_COMPLETED, ThreadPoolExecutor, wait as futures_wait,
)
from typing import (
    Dict, Iterator, List, Optional, Sequence, Tuple,
    Any, Union, Mapping, cast,
)
import collections.abc
import numpy as np
import dimod
import orjson
from dwave.cloud.computation import Future
from dwave.cloud.concurrency import Present
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave_topologies.embedding_loader import get_embedding_dict, embedding_exists
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology
# DefectInfo + the defect-clamp / array-reduction transforms live in shared/ (no
# D-Wave SDK) so the feeder's spawn workers can run prepare_reduced without
# importing this module. Re-exported here for back-compat (tests + base_miner
# import DefectInfo from QPU.dwave_sampler).
from shared.problem_prep import (  # noqa: F401  (DefectInfo re-exported)
    DefectInfo,
    ReducedProblem,
    clamp_fixed_variables,
    live_topology,
    prepare_reduced,
    rebuild_ising,
)

logger = logging.getLogger(__name__)

def _default_submit_workers(queue_depth: int) -> int:
    """Concurrent submit threads, scaled to the node.

    With the vectorized ``_submit_qp`` path the per-submit work is ~ms (encode +
    orjson + an enqueue-only ``client._submit``), not the old ~seconds of
    dimod/dict building, so only a little concurrency is needed to overlap it.
    Capped at ``cpu_count`` (never above ``queue_depth``) rather than the old
    ``cpu_count * 2``: on a 4-core node, fewer submit threads leave cores for the
    feeder driver's generator processes (which were starving — ``ready=0``,
    multi-second ``pop_blocking`` waits — while 8 submit threads thrashed).
    """
    return max(1, min(queue_depth, os.cpu_count() or 8))


def _wait_for_completions(futures, *, min_done, timeout):
    """Block until at least ``min_done`` of ``futures`` complete.

    Thin wrapper over :meth:`dwave.cloud.computation.Future.wait_multiple` so
    the streaming pump blocks on an event (releasing the GIL) instead of
    busy-polling ``future.done()`` — a spin loop here starves the dwave-client's
    single-threaded problem encoder on CPU-constrained nodes, serializing
    submission. Isolated as a module function so unit tests can substitute a
    ``done()``-poll over lightweight fake futures.

    Args:
        futures: Raw cloud Future objects to await.
        min_done: Return once this many have completed.
        timeout: Maximum seconds to block before returning (bounds stop-event
            responsiveness; does not spin).

    Returns:
        ``(done, remaining)`` lists, as ``Future.wait_multiple`` returns.
    """
    return Future.wait_multiple(futures, min_done=min_done, timeout=timeout)


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

# Type definitions to match base_miner
Variable = collections.abc.Hashable


class QPEncoder:
    """Vectorized D-Wave ``qp`` encoder for a fixed (solver, live-topology).

    Reproduces ``dwave.cloud.coders.encode_problem_as_qp`` **byte-for-byte** for
    the native (no-embedding) Ising path, but straight from the feeder's dense
    ``float64`` arrays — no per-submit Python dict / dimod object building. That
    object building (``rebuild_ising`` + ``from_ising`` + the SDK's dict
    extraction + its ~46k-term comprehensions, all GIL-held) was ~35ms+/submit on
    a fast box and stacked to ~11s across the submit threads on the slow node.
    This precomputes the encoding-order index maps once; :meth:`encode` is then a
    couple of vectorized numpy gathers + base64 (~0.5ms), so it no longer
    bottlenecks on CPU.

    The arrays come in ``live_nodes``/``live_edges`` order (the feeder's reduced
    layout); the wire wants ``solver._encoding_qubits``/``_encoding_couplers``
    order with NaN for inactive (non-live) qubits and 0 for live coupler pairs
    that aren't problem edges. ``test_qp_encoder`` asserts byte-identity against
    the SDK encoder so this can never silently diverge from consensus.
    """

    def __init__(self, encoding_qubits, encoding_couplers, live_nodes, live_edges):
        active = set(live_nodes)
        live_pos = {int(n): i for i, n in enumerate(live_nodes)}
        # lin: encoding-qubit order; active qubit -> its h_vec slot, else NaN.
        self._n_qubits = len(encoding_qubits)
        active_enc = [i for i, q in enumerate(encoding_qubits) if q in active]
        self._lin_dst = np.array(active_enc, dtype=np.intp)
        self._lin_src = np.array(
            [live_pos[int(encoding_qubits[i])] for i in active_enc], dtype=np.intp,
        )
        # quad: active-active encoding_couplers in order; problem edge -> its
        # j_vec slot, else -1 (encoded as 0.0). Matches the SDK's undirected get.
        edge_pos: Dict[Tuple[int, int], int] = {}
        for k, (u, v) in enumerate(live_edges):
            edge_pos[(int(u), int(v))] = k
            edge_pos[(int(v), int(u))] = k
        quad_src = [
            edge_pos.get((int(q1), int(q2)), -1)
            for (q1, q2) in encoding_couplers
            if q1 in active and q2 in active
        ]
        self._quad_src = np.array(quad_src, dtype=np.intp)
        self._n_quad = len(quad_src)
        self._quad_mask = self._quad_src >= 0

    def encode(self, h_vec: np.ndarray, j_vec: np.ndarray, offset: float = 0.0) -> dict:
        """Return the SAPI ``qp`` data dict for the given reduced arrays."""
        lin = np.full(self._n_qubits, np.nan, dtype="<f8")
        lin[self._lin_dst] = np.asarray(h_vec, dtype=np.float64)[self._lin_src]
        quad = np.zeros(self._n_quad, dtype="<f8")
        if self._n_quad:
            quad[self._quad_mask] = np.asarray(
                j_vec, dtype=np.float64
            )[self._quad_src[self._quad_mask]]
        return {
            "format": "qp",
            "lin": base64.b64encode(lin.astype("<f8").tobytes()).decode("utf-8"),
            "quad": base64.b64encode(quad.astype("<f8").tobytes()).decode("utf-8"),
            "offset": offset,
        }


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
        token: Optional[str] = None,
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
            token: Optional D-Wave API token. Passed verbatim to
                  `DWaveSampler(token=...)`. When unset the SDK falls
                  back to DWAVE_API_KEY env var → ~/.config/dwave/dwave.conf
                  → fails with "API token not defined". Honoring an
                  explicit kwarg lets a TOML `[dwave].token` value win
                  without requiring operators to also set the env var.
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

        # Token resolution order: explicit kwarg (TOML `[dwave].token`)
        # → DWAVE_API_KEY env var → SDK config file. Only warn when none
        # of those are present; the SDK itself will fail loudly at the
        # DWaveSampler() call below.
        if token:
            logger.debug(f"[QPU] using explicit D-Wave token (length: {len(token)})")
        else:
            api_key = os.environ.get('DWAVE_API_KEY')
            if not api_key:
                logger.warning(
                    "[QPU] no D-Wave token set (DWAVE_API_KEY env unset, "
                    "no token kwarg, no SDK config file)"
                )
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
            if token is not None:
                sampler_kwargs['token'] = token
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
            self._defective_edges: set = set()
            logger.info(f"[QPU] Embedding loaded: {len(embedding)} logical qubits mapped to hardware")

            # Use topology's graph directly
            self.nodelist: List[Variable] = topology.nodes
            self.edgelist: List[Tuple[Variable, Variable]] = topology.edges

        else:
            # Native hardware topology - no embedding needed
            logger.info("[QPU] Using native hardware topology (no embedding needed)")
            self.sampler = base_sampler
            self.embedding = None
            if topology is not None:
                # Detect defective qubits and couplers by comparing
                # stored topology against live QPU hardware.
                live_node_set = set(base_sampler.nodelist)
                stored_node_set = set(topology.nodes)

                self._defective_qubits: List[int] = sorted(
                    stored_node_set - live_node_set
                )
                extra_qubits = sorted(live_node_set - stored_node_set)

                # Detect defective edges (couplers offline between two
                # live nodes). Normalize direction for comparison.
                live_edge_set = {
                    (min(u, v), max(u, v))
                    for u, v in base_sampler.edgelist
                }
                defective_node_set = set(self._defective_qubits)
                self._defective_edges: set = set()
                for u, v in topology.edges:
                    if u in defective_node_set or v in defective_node_set:
                        continue  # handled by node clamping
                    key = (min(u, v), max(u, v))
                    if key not in live_edge_set:
                        self._defective_edges.add((u, v))

                if self._defective_qubits:
                    logger.warning(
                        f"[QPU] {len(self._defective_qubits)} defective "
                        f"qubits (offline on live QPU): "
                        f"{self._defective_qubits[:20]}"
                        f"{'...' if len(self._defective_qubits) > 20 else ''}"
                    )
                if self._defective_edges:
                    logger.warning(
                        f"[QPU] {len(self._defective_edges)} defective "
                        f"couplers (offline between live qubits)"
                    )
                if self._defective_qubits or self._defective_edges:
                    logger.warning("[QPU] Will use variable clamping")
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
                self._defective_edges: set = set()
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

        # Live (non-defective) topology ordering — the single source of truth
        # shared with the feeder workers (passed as prepare_reduced prep_args) so
        # a reduced-problem array position maps back to the same label on both
        # the producer (feeder) and consumer (submit) sides. Fixed for the QPU
        # session. With no defects these equal nodes/edges.
        self.live_nodes, self.live_edges = live_topology(
            self.nodes, self.edges, self._defective_qubits, self._defective_edges
        )

        # Fast-submit encoder for the native (no-embedding) path: precompute the
        # live->encoding-order maps so each streaming submit skips dimod/dict
        # building entirely. Only built when there's no embedding (the deployed
        # native-topology case) and the cloud solver exposes its encoding order;
        # otherwise _submit_prepared falls back to the dimod path.
        self._qp_encoder: Optional[QPEncoder] = None
        if self.embedding is None:
            solver = getattr(self.qpu_solver, "solver", None)
            enc_q = getattr(solver, "_encoding_qubits", None)
            enc_c = getattr(solver, "_encoding_couplers", None)
            if enc_q is not None and enc_c is not None:
                self._qp_encoder = QPEncoder(
                    enc_q, enc_c, self.live_nodes, self.live_edges,
                )

    def close(self):
        """Release QPU connection resources.

        Closes the D-Wave cloud client without waiting for in-flight jobs.
        Any pending futures from sample_ising_async() are abandoned.
        """
        if hasattr(self, 'qpu_solver'):
            # close(wait=False) tells the D-Wave client to shut down
            # immediately without waiting for in-flight jobs to complete.
            # Without this, the client's background submission thread
            # blocks indefinitely on atexit in multiprocessing workers.
            try:
                self.qpu_solver.client.close(wait=False)
            except Exception:
                pass

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
        nonce_seed: Union[int, bytes],
    ) -> Tuple[
        Dict[int, float],
        Dict[Tuple[int, int], float],
        Dict[int, int],
        float,
        Dict[Tuple[int, int], float],
    ]:
        """Clamp defective qubits to deterministic spins and adjust neighbors.

        For each offline qubit k, assigns a fixed spin s_k (deterministic from
        nonce_seed) and absorbs its coupling energy into neighbors' h-fields:
            h'[j] += J[k,j] * s_k  for all neighbors j of k

        This preserves the energy contribution of the clamped qubit in the
        reduced problem, so the QPU optimizes the remaining variables correctly.

        Args:
            h: Linear biases for all nodes (full topology).
            J: Quadratic biases for all edges (full topology).
            nonce_seed: Seed for deterministic spin assignment. Accepts either
                the 32-byte block nonce (post-MR-!20 wire shape) or a legacy
                int seed.

        Returns:
            5-tuple (h_reduced, J_reduced, fixed_spins, energy_offset, removed_edges):
            - h_reduced: biases without defective qubits (neighbors adjusted)
            - J_reduced: couplings without edges involving defective qubits
            - fixed_spins: {qubit_id: spin_value} for solution reconstruction
            - energy_offset: constant energy contribution from clamped qubits
              (h-fields and mutual J among clamped pairs); add once per sample
            - removed_edges: {(u, v): val} for live-qubit coupler pairs whose
              hardware coupler is offline; excluded from J_reduced and offset,
              computed per-sample in _reconstruct_full_sampleset

        Thin instance wrapper over :func:`shared.problem_prep.clamp_fixed_variables`
        (the SDK-free transform the feeder workers also call), binding this
        sampler's ``_defective_qubits``/``_defective_edges``.
        """
        return clamp_fixed_variables(
            h, J, nonce_seed, self._defective_qubits, self._defective_edges
        )

    @staticmethod
    def reconstruct_full_sampleset(
        reduced_sampleset: dimod.SampleSet,
        defect_info: DefectInfo,
    ) -> dimod.SampleSet:
        """Reconstruct full-topology sampleset from reduced QPU results.

        Inserts fixed spins and corrects energies using the precomputed
        offset + defective coupler contributions. Does NOT rebuild a BQM
        or recompute energies from scratch.

        Pure transform of ``(reduced_sampleset, defect_info)`` — it reads no
        instance/connection state, so it is a ``staticmethod``. This lets the
        connection-less worker miner reconstruct clamped samples without a
        live D-Wave sampler (the connection lives in the stream-driver
        process).

        Call this only for samplesets that contain promising candidates
        (QPU energy + offset < threshold). Most samplesets never need it.
        """
        fixed_spins = defect_info.fixed_spins
        offset = defect_info.energy_offset
        removed = defect_info.removed_edges
        has_removed = bool(removed)

        samples = []
        energies = []
        for datum in reduced_sampleset.data():
            full_sample = dict(datum.sample)
            full_sample.update(fixed_spins)

            corrected_energy = datum.energy + offset
            if has_removed:
                for (u, v), j_val in removed.items():
                    corrected_energy += j_val * full_sample[u] * full_sample[v]

            samples.append(full_sample)
            energies.append(corrected_energy)

        info = dict(reduced_sampleset.info) if hasattr(reduced_sampleset, 'info') else {}

        return dimod.SampleSet.from_samples(
            samples, vartype=dimod.SPIN, energy=energies, info=info,
        )

    def _prepare_defect_handling(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        kwargs: Dict[str, Any],
    ) -> Tuple[
        Union[Mapping[Variable, float], Sequence[float]],
        Mapping[Tuple[Variable, Variable], float],
        Optional[DefectInfo],
    ]:
        """Pop ``nonce_seed`` and clamp defective qubits if any are present.

        Returns ``(h_eff, J_eff, defect_info)`` where the effective problem is
        the defect-reduced one (and ``defect_info`` is populated) when defects
        exist and a seed was supplied; otherwise the inputs are returned
        unchanged with ``defect_info`` set to ``None``.
        """
        nonce_seed = kwargs.pop('nonce_seed', None)

        has_defects = self._defective_qubits or self._defective_edges
        if not (has_defects and nonce_seed is not None):
            return h, J, None

        h_dict = dict(h) if not isinstance(h, dict) else h
        J_dict = dict(J) if not isinstance(J, dict) else J
        h_reduced, J_reduced, fixed_spins, offset, removed = (
            self._clamp_defective_qubits(h_dict, J_dict, nonce_seed)
        )
        return h_reduced, J_reduced, DefectInfo(fixed_spins, offset, removed)

    @staticmethod
    def reconstruct_full_matrix(
        reduced_sample: np.ndarray,
        reduced_energy: np.ndarray,
        defect_info: DefectInfo,
        nodes: Sequence[Variable],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Positionally reconstruct a label-less reduced sample matrix.

        The streaming path is positional, not labeled: the stream driver writes
        a raw ``int8`` matrix to shared memory and discards dimod's variable
        labels. dimod orders a ``SampleSet``'s columns by **sorted variable
        label**, so the reduced matrix's columns are the live (non-clamped)
        node labels in ascending order — NOT in ``nodes`` order. The output, by
        contrast, must be in ``nodes`` order because the consensus energy
        recompute (``energies_for_solutions``) reads column ``pos`` as
        ``nodes[pos]``. This maps the sorted live columns back to their
        ``nodes`` positions explicitly, so correctness holds for any node
        ordering rather than only when ``nodes`` happens to be sorted.

        Clamped spins are reinserted at their topology positions and each
        energy is corrected by the constant offset plus the per-sample
        defective-coupler contributions.

        Unlike :meth:`reconstruct_full_sampleset` (which reads dimod's
        ``.data()`` and is used by the synchronous, labeled ``sample_ising``
        path), this consumes raw arrays so it works on the connection-less
        worker's zero-copy ring views. The energy correction MUST stay
        consensus-identical to that method — validators recompute energy on the
        full topology.

        Args:
            reduced_sample: ``int8`` ``(R, n_live)`` matrix; columns are the
                live (non-clamped) node labels in ascending (dimod-sorted)
                order.
            reduced_energy: ``float64`` ``(R,)`` reduced-problem QPU energies.
            defect_info: :class:`DefectInfo` with ``fixed_spins``,
                ``energy_offset``, and ``removed_edges``.
            nodes: Full topology node order; the column order of the output.

        Returns:
            ``(full_sample, full_energy)``: ``int8`` ``(R, len(nodes))`` matrix
            in ``nodes`` order with clamped spins reinserted, and ``float64``
            ``(R,)`` energies corrected to the full topology.

        Raises:
            ValueError: if the reduced width disagrees with the live-node count,
                or if a ``fixed_spins`` / ``removed_edges`` label is absent from
                ``nodes`` — each is a topology mismatch that would otherwise
                silently scatter spins or drop a clamp and corrupt the
                consensus energy.
        """
        fixed_spins = defect_info.fixed_spins
        reduced = np.asarray(reduced_sample, dtype=np.int8)
        n_rows, n_live = reduced.shape

        missing_clamps = set(fixed_spins) - set(nodes)
        if missing_clamps:
            raise ValueError(
                f"fixed_spins labels {sorted(missing_clamps)} absent from nodes "
                f"— topology mismatch; clamped spins would be silently dropped"
            )
        # dimod sorts SampleSet columns by label, so the reduced columns are the
        # live labels ascending. Map each live label to its reduced column.
        live_labels = sorted(n for n in nodes if n not in fixed_spins)
        if n_live != len(live_labels):
            raise ValueError(
                f"reduced width {n_live} != live nodes {len(live_labels)} "
                f"({len(nodes)} topology - {len(fixed_spins)} clamped)"
            )
        reduced_col = {label: i for i, label in enumerate(live_labels)}

        full = np.empty((n_rows, len(nodes)), dtype=np.int8)
        pos: Dict[Variable, int] = {}  # node label -> output column index
        for out_col, node in enumerate(nodes):
            pos[node] = out_col
            if node in fixed_spins:
                full[:, out_col] = fixed_spins[node]
            else:
                full[:, out_col] = reduced[:, reduced_col[node]]

        energy = np.asarray(reduced_energy, dtype=np.float64) + defect_info.energy_offset
        for (u, v), j_val in defect_info.removed_edges.items():
            if u not in pos or v not in pos:
                raise ValueError(
                    f"removed_edge endpoint ({u}, {v}) absent from nodes — "
                    f"topology mismatch on the consensus energy path"
                )
            energy = energy + j_val * (
                full[:, pos[u]].astype(np.float64) * full[:, pos[v]]
            )
        return full, energy

    def sample_ising(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> dimod.SampleSet:
        """Sample from the D-Wave QPU (synchronous, with full reconstruction).

        For the streaming path, use sample_ising_async + lazy reconstruction
        instead — it avoids reconstructing samples that don't meet threshold.
        This method always reconstructs for backward compatibility.
        """
        h_eff, J_eff, defect_info = self._prepare_defect_handling(h, J, kwargs)
        sampleset = self._sample_ising_inner(h_eff, J_eff, **kwargs)
        if defect_info is not None:
            return self.reconstruct_full_sampleset(sampleset, defect_info)
        return sampleset

    def _chain_strength(self, bqm: dimod.BinaryQuadraticModel, multiplier: float) -> float:
        """Compute chain strength as the largest absolute bias scaled by *multiplier*.

        If the BQM has quadratic interactions the largest |J| is used; otherwise
        the largest |h| is used.  Falls back to 1.0 when the BQM is empty.
        """
        if bqm.num_interactions > 0:
            return max(abs(b) for b in bqm.quadratic.values()) * multiplier
        return max(abs(b) for b in bqm.linear.values()) * multiplier if bqm.linear else 1.0

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
                logger.warning(
                    "BQM variables don't match embedding keys! "
                    "BQM vars: %d, range: %d-%d; Embedding vars: %d, range: %d-%d",
                    len(bqm_vars), min(bqm_vars), max(bqm_vars),
                    len(embedding_vars), min(embedding_vars), max(embedding_vars),
                )

            # Calculate chain strength explicitly so we control the multiplier
            chain_strength = self._chain_strength(bqm, chain_strength_multiplier)

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
                logger.warning(
                    "Sampleset variables don't match logical topology! "
                    "Expected: %d vars (0-%d); Got: %d vars (%d-%d); "
                    "Missing: %s; Extra: %s",
                    len(expected_vars), max(expected_vars),
                    len(actual_vars), min(actual_vars), max(actual_vars),
                    sorted(list(expected_vars - actual_vars))[:20],
                    sorted(list(actual_vars - expected_vars))[:20],
                )

        return sampleset

    def sample_ising_async(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> Tuple[Any, Optional[DefectInfo]]:
        """Submit Ising problem to QPU and return (future, defect_info).

        The future resolves to a REDUCED sampleset (defective qubits
        stripped). The caller uses defect_info to screen candidates by
        energy (QPU_energy + offset < threshold) and only reconstructs
        the full-topology sampleset for winners.

        Returns:
            (future, defect_info) where defect_info is None if no defects.
        """
        h_eff, J_eff, defect_info = self._prepare_defect_handling(h, J, kwargs)
        future = self._sample_ising_async_inner(h_eff, J_eff, **kwargs)
        return future, defect_info

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
            chain_strength = self._chain_strength(source_bqm, chain_strength_multiplier)

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

    def _submit_prepared(
        self,
        h_vec: np.ndarray,
        j_vec: np.ndarray,
        **kwargs,
    ) -> Union['Future', EmbeddedFuture]:
        """Lean streaming submit for a feeder-reduced problem.

        The defect-clamp + array reduction already happened in the feeder
        workers (off this path); this only rebuilds the dimod model from the
        ``ProblemView`` arrays and dispatches ``sample_bqm``. The reduced arrays
        are laid out in ``self.live_nodes``/``self.live_edges`` order — the same
        ordering the feeder used (both from ``live_topology``) — so the rebuilt
        labels are exact and the consensus-energy reconstruction is unaffected.

        Args:
            h_vec: float64 reduced linear biases (``ProblemView`` h vector).
            j_vec: float64 reduced couplings (``ProblemView`` j vector).
            **kwargs: forwarded to ``solver.sample_bqm`` (num_reads, label, …).

        Returns:
            A raw cloud ``Future`` (no embedding) or an :class:`EmbeddedFuture`.
        """
        if 'label' not in kwargs:
            kwargs['label'] = self.job_label
        chain_strength_multiplier = kwargs.pop('chain_strength_multiplier', 1.5)

        # Fast path (native topology): encode the reduced arrays straight to the
        # SAPI wire and submit — no dimod/dict object building (the GIL-bound
        # ~35ms+/submit that stacked to ~11s across submit threads on the slow
        # node). Byte-identical to the dimod path (see QPEncoder / test_qp_encoder).
        if self.embedding is None and self._qp_encoder is not None:
            return self._submit_qp(h_vec, j_vec, **kwargs)

        h, J = rebuild_ising(h_vec, j_vec, self.live_nodes, self.live_edges)
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        if self.embedding is not None:
            chain_strength = self._chain_strength(bqm, chain_strength_multiplier)
            target_bqm = embed_bqm(
                bqm, self.embedding, self.qpu_solver.adjacency,
                chain_strength=chain_strength,
            )
            raw_future = self.qpu_solver.solver.sample_bqm(target_bqm, **kwargs)
            return EmbeddedFuture(
                future=raw_future,
                source_bqm=bqm,
                embedding=self.embedding,
                chain_strength=chain_strength,
            )
        return self.qpu_solver.solver.sample_bqm(bqm, **kwargs)

    def _submit_qp(
        self,
        h_vec: np.ndarray,
        j_vec: np.ndarray,
        *,
        num_reads: int,
        answer_mode: str = "raw",
        annealing_time: Optional[float] = None,
        label: Optional[str] = None,
        offset: float = 0.0,
        **extra: Any,
    ) -> 'Future':
        """Submit a reduced problem via the vectorized ``qp`` encoder.

        Replicates ``StructuredSolver._sample`` (body + params + Future +
        ``client._submit``) but with :class:`QPEncoder` producing the ``data``
        dict from numpy arrays — skipping ``check_problem`` and all dict/dimod
        building. ``client._submit`` only enqueues, so this returns in ~ms. The
        offset stays 0: the defect-clamp energy offset is applied later during
        reconstruction (it never goes on the wire), matching the dimod path.

        Returns:
            The cloud :class:`~dwave.cloud.computation.Future` (raw, no embedding
            — this path runs only when ``self.embedding is None``).
        """
        solver = self.qpu_solver.solver
        params: Dict[str, Any] = {"num_reads": int(num_reads), "answer_mode": answer_mode}
        if annealing_time is not None:
            params["annealing_time"] = annealing_time
        params.update(extra)
        combined = dict(solver._params)
        combined.update(params)
        solver._format_params("ising", combined)

        body_dict: Dict[str, Any] = {
            "solver": solver.identity.dict(),
            "data": self._qp_encoder.encode(h_vec, j_vec, offset),
            "type": "ising",
            "params": combined,
        }
        if label is not None:
            body_dict["label"] = label
        body_data = orjson.dumps(body_dict, option=orjson.OPT_SERIALIZE_NUMPY)
        body = Present(result=body_data)
        computation = Future(
            solver=solver, id_=None, return_matrix=solver.return_matrix,
        )
        computation._offset = offset
        solver.client._submit(body, computation)
        return computation

    def sample_ising_streaming(
        self,
        models: Any,
        *,
        num_reads: int,
        num_sweeps: int = 0,
        queue_depth: int = 30,
        annealing_time: Optional[float] = None,
        energy_threshold_milli: int = 0,
        stop_event: Optional[Any] = None,
        **_kw: Any,
    ) -> Iterator[Tuple[Any, Any]]:
        """Async streaming pump: keep queue_depth submissions in flight; yield
        (model, raw_reduced_sampleset) with sampleset.info['defect_info'] set.

        No reconstruction gating — the consumer reconstructs survivors.
        Cancels in-flight futures on GeneratorExit (the generic StreamContext
        closes this generator on a chain-head switch).

        Args:
            models: Feeder providing IsingModel objects via ``pop_blocking()``
                or ``__next__``.
            num_reads: QPU reads per submission.
            num_sweeps: Unused (accepted for interface compatibility with the
                generic ``StreamContext``).
            queue_depth: Number of concurrent in-flight QPU jobs.
            annealing_time: Anneal duration in microseconds; ``None`` uses the
                QPU default.
            energy_threshold_milli: Accepted for interface compatibility; not
                used (no gating in this path — the consumer gates).
            stop_event: When set, stop submitting and cancel in-flight work.
            **_kw: Extra kwargs accepted silently for interface compatibility.

        Yields:
            ``(model, raw_reduced_sampleset)`` in completion order, with
            ``sampleset.info['defect_info']`` set to the
            :class:`DefectInfo` (or ``None`` when no defects).
        """
        # pending: {id(future): (model, future, defect_info, job_index)}
        pending: Dict[int, Tuple[Any, Any, Optional[DefectInfo], int]] = {}
        job_index: int = 0
        feeder_exhausted: bool = False
        # Submission-cost profiling: how long sample_ising_async takes to
        # return. The pump's fill loop calls it serially, so if it blocks (a
        # synchronous encode/upload in the SDK), submission serializes here
        # regardless of queue_depth. ~tens of ms = async handoff (healthy);
        # seconds = the fill loop is the bottleneck.
        submit_total_s: float = 0.0
        submit_n: int = 0
        submit_max_s: float = 0.0
        # Decode-cost profiling: ``future.sampleset`` parses the QPU wire
        # response into a SampleSet inline on this (single-GIL) thread, the same
        # thread that pops+unpickles feeder models and harvests submissions. If
        # decode dominates, it — not generation — is what pegs the stream-driver
        # core and starves the feeder's result pipe. Reported as decode_mean.
        decode_total_s: float = 0.0
        decode_n: int = 0

        def _stopped() -> bool:
            return stop_event is not None and stop_event.is_set()

        def _best_effort_cancel_future(fut: Any, fidx: int) -> None:
            cancel_fn = getattr(fut, "cancel", None)
            if not callable(cancel_fn):
                return
            try:
                cancel_fn()
            except Exception as exc:  # noqa: BLE001 — advisory; log and continue
                logger.debug(
                    "D-Wave future.cancel() failed for job %d (best-effort): "
                    "%s: %s",
                    fidx, type(exc).__name__, exc,
                )

        def _cancel_all() -> None:
            """Best-effort cancel every in-flight future and clear pending."""
            for _mdl, fut, _d, fidx in list(pending.values()):
                _best_effort_cancel_future(fut, fidx)
            pending.clear()

        # Submission tasks running on the pool (each yields one QPU future).
        submitting: set = set()

        def _pop_model() -> Optional[Any]:
            """Pop one model in the main thread; None when the feeder drained."""
            nonlocal feeder_exhausted
            pop = getattr(models, "pop_blocking", None)
            try:
                return pop() if callable(pop) else next(models)  # type: ignore[call-overload]
            except StopIteration:
                feeder_exhausted = True
                return None

        def _submit_job(model: Any, idx: int):
            """Submit one feeder-reduced problem on a pool thread.

            ``model`` is a :class:`~shared.problem_prep.ReducedProblem`: the
            defect-clamp + array reduction already ran in the feeder workers, so
            this only rebuilds + dispatches. Returns
            ``(model, qpu_future, defect_info, idx, submit_seconds)``.
            """
            kw: Dict[str, Any] = {
                "num_reads": num_reads,
                "answer_mode": "raw",
                "label": f"{self.job_label}_s{idx}",
            }
            if annealing_time is not None:
                kw["annealing_time"] = annealing_time
            t = time.monotonic()
            future = self._submit_prepared(model.h_vec, model.j_vec, **kw)
            return model, future, model.defect_info, idx, time.monotonic() - t

        def _harvest_submissions() -> None:
            """Move finished submission tasks into pending; record submit cost."""
            nonlocal submit_total_s, submit_n, submit_max_s
            for tf in [t for t in submitting if t.done()]:
                submitting.discard(tf)
                if tf.cancelled():
                    continue
                try:
                    model, future, defect_info, _idx, dt = tf.result()
                except Exception as exc:  # noqa: BLE001 — one bad submit must not kill the pump
                    logger.warning("QPU submission failed (dropped): %s", exc)
                    continue
                pending[id(future)] = (model, future, defect_info, _idx)
                submit_total_s += dt
                submit_n += 1
                submit_max_s = max(submit_max_s, dt)

        def _drain_submissions() -> None:
            """On teardown, cancel queued submits and cancel any QPU futures the
            running ones already produced (a running submit can't be interrupted,
            so harvest + cancel its future to avoid leaking a submitted problem).
            """
            for tf in list(submitting):
                tf.cancel()
            futures_wait(submitting, timeout=2.0)
            for tf in submitting:
                if tf.done() and not tf.cancelled():
                    try:
                        _m, qf, _d, _i, _dt = tf.result()
                        _best_effort_cancel_future(qf, -1)
                    except Exception:  # noqa: BLE001 — best-effort cleanup
                        pass

        def _emit_diagnostic() -> None:
            pop_stats = getattr(models, "stats", None)
            if not callable(pop_stats):
                return
            fstats = pop_stats()
            logger.info(
                "[QPU] stream depth: in_flight=%d/%d submitting=%d "
                "feeder_ready=%d/%d drained=%d wait_total=%.2fs "
                "submit_mean=%.0fms submit_max=%.0fms decode_mean=%.0fms",
                len(pending), queue_depth, len(submitting),
                fstats.get("ready", 0),
                fstats.get("buffer_size", 0),
                fstats.get("drained_count", 0),
                fstats.get("pop_wait_total_s", 0.0),
                (submit_total_s / submit_n * 1000.0) if submit_n else 0.0,
                submit_max_s * 1000.0,
                (decode_total_s / decode_n * 1000.0) if decode_n else 0.0,
            )

        submit_workers = _default_submit_workers(queue_depth)
        submit_pool = ThreadPoolExecutor(
            max_workers=submit_workers, thread_name_prefix="qpu-submit",
        )
        try:
            while not _stopped():
                # Keep queue_depth problems in flight, counting both already
                # submitted (pending) and in-progress submissions (submitting).
                while (
                    len(pending) + len(submitting) < queue_depth
                    and not _stopped()
                    and not feeder_exhausted
                ):
                    model = _pop_model()
                    if model is None:
                        break
                    submitting.add(submit_pool.submit(_submit_job, model, job_index))
                    job_index += 1

                _harvest_submissions()

                if not pending and not submitting:
                    if feeder_exhausted:
                        break
                    continue
                if not pending:
                    # Only submissions in flight — wait for one to land, then
                    # loop to harvest it (no QPU futures to wait on yet).
                    futures_wait(submitting, timeout=0.5, return_when=FIRST_COMPLETED)
                    continue

                # Wait (GIL-released) for the next QPU completion. ``pending``
                # keys on the (possibly EmbeddedFuture) wrapper, but
                # wait_multiple needs the raw cloud Future, so map raw -> key.
                raw_to_pending: Dict[int, int] = {}
                raw_futures = []
                for fid, (_, fut, _, _) in pending.items():
                    raw = getattr(fut, "_future", fut)
                    raw_to_pending[id(raw)] = fid
                    raw_futures.append(raw)
                done, _remaining = _wait_for_completions(
                    raw_futures, min_done=1, timeout=0.5,
                )
                if _stopped():
                    break
                for finished in done:
                    completed_id = raw_to_pending.get(id(finished))
                    if completed_id is None:
                        continue
                    model, future, defect_info, _ = pending.pop(completed_id)
                    _t_dec = time.monotonic()
                    raw_ss = future.sampleset
                    decode_total_s += time.monotonic() - _t_dec
                    decode_n += 1
                    # Attach defect_info so the consumer can reconstruct survivors.
                    raw_ss.info["defect_info"] = defect_info
                    if job_index % 50 == 0:
                        _emit_diagnostic()
                    yield model, raw_ss
        finally:
            _cancel_all()
            _drain_submissions()
            submit_pool.shutdown(wait=False)