"""IBM QAOA miner for quantum blockchain mining.

Analogous to dwave_miner.py — uses QAOASolverWrapper to solve Ising problems
generated from block headers.  Unlike the D-Wave miner's streaming queue,
this uses a sequential loop with one QAOA solve at a time, since each QAOA
solve is already an expensive iterative process.

The miner submits QAOA jobs asynchronously and polls for completion, allowing
it to react to stop_event (another node won the block) and difficulty decay
between solves.
"""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from typing import Optional, Dict, Tuple, Any, List

init_logger = logging.getLogger(__name__)

from QPU.IBM.ibm_qaoa_solver import QAOASolverWrapper, QAOAFuture
from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)
from shared.block_requirements import compute_current_requirements
from shared.energy_utils import energy_to_difficulty


class IBMQAOAMiner(BaseMiner):
    """Miner that uses IBM QAOA to solve quantum proof-of-work problems.

    Architecture:
        - Sequential solve loop (one problem at a time)
        - Async submission with polling for stop_event responsiveness
        - Difficulty decay re-evaluation between solves
        - Cached attempts re-checked against relaxed difficulty
    """

    def __init__(
        self,
        miner_id: str,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        backend: Optional[Any] = None,
        p: int = 1,
        optimizer: str = 'COBYLA',
        shots: int = 1024,
        final_shots: Optional[int] = None,
        max_iter: Optional[int] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
        **cfg,
    ):
        """
        Initialize IBM QAOA miner.

        Args:
            miner_id: Unique identifier for this miner.
            nodes: Logical node list for the Ising topology.
            edges: Logical edge list for the Ising topology.
            backend: Qiskit backend.  None → AerSimulator (local simulation).
            p: QAOA circuit depth (number of layers).
            optimizer: Classical optimizer — one of 'COBYLA', 'NELDER_MEAD',
                      'POWELL', 'L_BFGS_B', 'SPSA'.
            shots: Shots per evaluation during QAOA optimization loop.
            final_shots: Shots for the final sampling run.  Defaults to 4×shots.
            max_iter: Override max optimizer iterations.
            optimizer_options: Additional optimizer-specific overrides.
            **cfg: Reserved for future configuration.
        """
        init_logger.info(
            f"[QAOA] Initializing IBMQAOAMiner: {len(nodes)} nodes, "
            f"{len(edges)} edges, p={p}, optimizer={optimizer}"
        )

        try:
            solver = QAOASolverWrapper(
                nodes=nodes,
                edges=edges,
                backend=backend,
                p=p,
                optimizer=optimizer,
                shots=shots,
                final_shots=final_shots,
                max_iter=max_iter,
                optimizer_options=optimizer_options,
            )
            init_logger.info("[QAOA] Solver ready")
        except Exception as e:
            init_logger.error(f"[QAOA] Failed to initialize solver: {e}")
            raise

        super().__init__(miner_id, solver, miner_type="IBM_QAOA")
        self.sampler: QAOASolverWrapper = solver  # narrow type for QAOA-specific methods
        self.miner_type = "IBM_QAOA"

        # Current async job (at most one in-flight at a time)
        self._current_future: Optional[QAOAFuture] = None

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful shutdown."""
        if hasattr(self, 'logger'):
            self.logger.info(
                f"QAOA miner {self.miner_id} received SIGTERM, cleaning up..."
            )

        try:
            # Cancel in-flight future
            if self._current_future is not None:
                self._current_future.cancel()
                self._current_future = None

            # Clear cached attempts
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during QAOA miner cleanup: {e}")

        sys.exit(0)

    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using IBM QAOA.

        Uses a sequential loop: generate Ising problem → submit QAOA solve
        asynchronously → poll for completion (checking stop_event and
        difficulty decay every 200ms) → evaluate result → repeat if not valid.

        During each solve, the poll loop actively checks for difficulty decay
        and re-evaluates cached attempts against relaxed thresholds.  If a
        cached attempt now qualifies, the in-progress solve is cancelled and
        the cached result is returned immediately.

        Args:
            prev_block: Previous block in the chain.
            node_info: Node information containing miner_id.
            requirements: NextBlockRequirements with difficulty settings.
            prev_timestamp: Timestamp from the previous block header.
            stop_event: Multiprocessing event to signal stop.

        Returns:
            MiningResult if successful, None if stopped or failed.
        """
        self.mining = True
        progress = 0
        self.top_attempts = []
        start_time = time.time()

        self.logger.debug(f"requirements: {requirements}")

        cur_index = prev_block.header.index + 1
        self.current_round_attempted = True

        # Apply difficulty decay
        current_requirements = compute_current_requirements(
            requirements, prev_timestamp, self.logger
        )

        nodes = self.sampler.nodes
        edges = self.sampler.edges

        params = adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )
        self.logger.info(
            f"[Block-{cur_index}] Mining with QAOA (p={params['p']}, "
            f"shots={params['shots']}, final_shots={params['final_shots']}, "
            f"max_iter={params['max_iter']})"
        )

        # ------- Sequential solve loop -------
        while not stop_event.is_set():

            # --- Generate a new Ising problem ---
            salt = random.randbytes(32)
            nonce = ising_nonce_from_block(
                prev_block.hash, node_info.miner_id, cur_index, salt
            )
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            self.current_stage = 'sampling'
            self.current_stage_start = time.time()

            # --- Submit QAOA solve (async for stop_event responsiveness) ---
            self.logger.info(
                f"[Block-{cur_index}] Attempt {progress + 1}: "
                f"nonce={nonce}, salt={salt.hex()[:8]}..."
            )

            self._current_future = self.sampler.solve_ising_async(
                h, J, stop_event=stop_event, params=params
            )
            future = self._current_future  # local ref avoids Optional checks

            # --- Poll for completion, checking difficulty decay while waiting ---
            # This is where responsiveness lives: every 200ms we check
            # stop_event AND re-evaluate difficulty.  If a cached attempt
            # now meets relaxed difficulty, we return it immediately and
            # cancel the in-progress solve instead of waiting for it.
            while not stop_event.is_set():
                if future.done():
                    break

                # Check for difficulty decay during the solve
                updated_requirements = compute_current_requirements(
                    requirements, prev_timestamp, self.logger
                )
                if current_requirements != updated_requirements:
                    current_requirements = updated_requirements
                    params = adapt_parameters(
                        current_requirements.difficulty_energy,
                        current_requirements.min_diversity,
                        current_requirements.min_solutions,
                        num_nodes=len(nodes),
                        num_edges=len(edges),
                    )
                    self.logger.info(
                        f"{self.miner_id} - Updated params due to difficulty "
                        f"decay: {params}"
                    )

                    # Re-check cached attempts against relaxed difficulty
                    for sample in self.top_attempts:
                        if (
                            min(sample.sampleset.record.energy)
                            <= current_requirements.difficulty_energy
                        ):
                            result = self.evaluate_sampleset(
                                sample.sampleset,
                                current_requirements,
                                nodes,
                                edges,
                                sample.nonce,
                                sample.salt,
                                prev_timestamp,
                                start_time,
                            )
                            if result:
                                self.logger.info(
                                    f"[Block-{cur_index}] Cached result now "
                                    f"meets decayed difficulty! Cancelling "
                                    f"in-progress solve."
                                )
                                future.cancel()
                                return result

                time.sleep(0.2)  # 200ms polling interval

            # If stopped externally, cancel and exit
            if stop_event.is_set():
                if not future.done():
                    future.cancel()
                self.logger.info("Mining stopped by external signal")
                return None

            # --- Process result ---
            progress += 1
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start

            try:
                sampleset = future.sampleset

                if sampleset is None:
                    # Solve was interrupted
                    self.logger.debug("QAOA solve returned None (interrupted)")
                    continue

                # Timing stats
                sample_time = future.elapsed
                self.timing_stats['sampling'].append(sample_time * 1e6)

                all_energies = sampleset.record.energy
                self.timing_stats['total_samples'] += len(all_energies)
                self.timing_stats['blocks_attempted'] += 1

                # Evaluate against requirements
                result = self.evaluate_sampleset(
                    sampleset,
                    current_requirements,
                    nodes,
                    edges,
                    nonce,
                    salt,
                    prev_timestamp,
                    start_time,
                )

                self.logger.debug(
                    f"QAOA result evaluated in "
                    f"{time.time() - postprocess_start:.2f}s"
                )
                self.timing_stats['postprocessing'].append(
                    (time.time() - postprocess_start) * 1e6
                )

                if result:
                    self.logger.info(
                        f"[Block-{cur_index}] Mined! Nonce: {nonce}, "
                        f"Salt: {salt.hex()[:4]}..., "
                        f"Energy: {result.energy:.2f}, "
                        f"Solutions: {result.num_valid}, "
                        f"Diversity: {result.diversity:.3f}, "
                        f"Attempts: {progress}, "
                        f"Time: {time.time() - start_time:.2f}s"
                    )
                    return result

                # Not valid — cache for potential difficulty decay re-check
                self.update_top_samples(
                    sampleset, nonce, salt, current_requirements
                )

                # Log progress
                if self.top_attempts:
                    best_energy = min(
                        self.top_attempts[0].sampleset.record.energy
                    )
                    self.logger.info(
                        f"Progress: {progress} solves, "
                        f"best energy: {best_energy:.2f}, "
                        f"solve time: {sample_time:.2f}s"
                    )

            except Exception as e:
                self.logger.error(f"Error processing QAOA result: {e}")

        self.logger.info("Stopping mining, no valid results found")
        return None


# ---------------------------------------------------------------------------
# Adaptive parameter tuning
# ---------------------------------------------------------------------------

def adapt_parameters(
    difficulty_energy: float,
    min_diversity: float,
    min_solutions: int,
    num_nodes: int,
    num_edges: int,
) -> Dict[str, Any]:
    """Calculate adaptive QAOA parameters based on difficulty.

    QAOA strategy differs from D-Wave:
      - Harder problems → deeper circuit (higher p)
      - Harder problems → more optimizer iterations
      - Harder problems → more final shots for solution diversity

    Args:
        difficulty_energy: Target energy threshold.
        min_diversity: Minimum solution diversity required.
        min_solutions: Minimum valid solutions required.
        num_nodes: Number of nodes in topology.
        num_edges: Number of edges in topology.

    Returns:
        Dictionary with QAOA-specific parameters.
    """
    difficulty = energy_to_difficulty(
        difficulty_energy,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )

    # QAOA depth: p=1 for easy, up to p=3 for hardest
    # Deeper circuits explore more of the energy landscape
    if difficulty < 0.3:
        p = 1
    elif difficulty < 0.7:
        p = 2
    else:
        p = 3

    # Optimizer iterations: scale with difficulty
    min_iter = 50
    max_iter = 200
    max_iter_val = int(min_iter + difficulty * (max_iter - min_iter))

    # Shots per optimizer evaluation: more shots = less noisy gradient
    min_shots = 512
    max_shots = 2048
    shots = int(min_shots + difficulty * (max_shots - min_shots))

    # Final sampling shots: need enough for min_solutions diverse results
    # Scale with both difficulty and min_solutions requirement
    base_final = max(min_solutions * 8, 1024)
    final_shots = int(base_final + difficulty * base_final)

    return {
        'p': p,
        'max_iter': max_iter_val,
        'shots': shots,
        'final_shots': final_shots,
    }