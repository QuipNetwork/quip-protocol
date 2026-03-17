"""CPU miner with greedy descent pre-filtering.

Scores batches of nonces cheaply via greedy descent, then runs
full SA only on the most promising candidates.
"""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import time
from typing import Optional

from shared.block_requirements import compute_current_requirements
from shared.miner_types import MiningResult
from shared.nonce_prefilter import batch_score_nonces
from CPU.sa_miner import SimulatedAnnealingMiner


class SAFilteredMiner(SimulatedAnnealingMiner):
    """CPU SA miner that pre-filters nonces via greedy descent.

    Instead of trying one nonce at a time, scores a batch of nonces
    cheaply and only runs full SA on the top candidates.
    """

    # Experiment tuning constants
    PREFILTER_BATCH_SIZE: int = 16
    PREFILTER_KEEP: int = 4
    PREFILTER_NUM_PASSES: int = 3
    PREFILTER_NUM_STARTS: int = 4

    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> Optional[MiningResult]:
        """Mine using batch pre-filtering + SA on top candidates."""
        # -- setup (mirrors BaseMiner.mine_block) --------------------------
        self.mining = True
        progress = 0
        self.top_attempts = []
        start_time = time.time()

        cur_index = prev_block.header.index + 1
        self.current_round_attempted = True
        self.logger.info(f"Mining block {cur_index} (filtered)...")

        if not self._pre_mine_setup(
            prev_block, node_info, requirements,
            prev_timestamp, stop_event, **kwargs,
        ):
            return None

        current_requirements = compute_current_requirements(
            requirements, prev_timestamp, self.logger,
        )

        nodes = self.sampler.nodes
        edges = self.sampler.edges

        params = self._adapt_mining_params(
            current_requirements, nodes, edges,
        )
        self.logger.info(f"{self.miner_id} - Adaptive params: {params}")

        current_num_sweeps = params.get('num_sweeps', 64)
        num_reads = params.get('num_reads', 100)
        max_num_sweeps = current_num_sweeps
        increment_interval = 30.0
        last_increment_time = start_time

        # -- main mining loop (batch pre-filter) ---------------------------
        while self.mining and not stop_event.is_set():
            # Sweep ramp-up
            current_time = time.time()
            if current_time - last_increment_time >= increment_interval:
                current_num_sweeps = min(
                    max_num_sweeps, int(current_num_sweeps * 1.05),
                )
                last_increment_time = current_time

            # Difficulty decay check
            updated = compute_current_requirements(
                requirements, prev_timestamp, self.logger,
            )
            if current_requirements != updated:
                current_requirements = updated
                params = self._adapt_mining_params(
                    current_requirements, nodes, edges,
                )
                self.logger.info(
                    f"{self.miner_id} - updated adaptive params: {params}",
                )

                # Check cached top attempts against new requirements
                for sample in self.top_attempts:
                    best_e = min(sample.sampleset.record.energy)
                    if best_e <= current_requirements.difficulty_energy:
                        result = self.evaluate_sampleset(
                            sample.sampleset, current_requirements,
                            nodes, edges, sample.nonce, sample.salt,
                            prev_timestamp, start_time,
                        )
                        if result:
                            self.logger.info(
                                f"[Block-{cur_index}] Already Mined! "
                                f"Energy: {result.energy:.2f}, "
                                f"Time: {time.time() - start_time:.2f}s",
                            )
                            self._post_mine_cleanup()
                            return result

            if stop_event.is_set():
                break

            # -- BATCH SCORING --
            prefilter_start = time.time()
            candidates = batch_score_nonces(
                prev_block.hash, node_info.miner_id, cur_index,
                nodes, edges,
                batch_size=self.PREFILTER_BATCH_SIZE,
                keep=self.PREFILTER_KEEP,
            )
            prefilter_time = time.time() - prefilter_start

            top_candidates = candidates
            self.logger.debug(
                f"Prefilter: {self.PREFILTER_BATCH_SIZE} nonces in "
                f"{prefilter_time * 1000:.1f}ms, "
                f"best greedy={candidates[0][4]:.0f}, "
                f"worst greedy={candidates[-1][4]:.0f}",
            )

            # -- SA on top candidates --
            for salt, nonce, h, J, greedy_e in top_candidates:
                if stop_event.is_set():
                    break

                self.current_stage = 'preprocessing'
                self.current_stage_start = time.time()

                try:
                    sample_start = time.time()
                    self.current_stage = 'sampling'
                    self.current_stage_start = sample_start

                    sampleset = self._sample(
                        h, J,
                        num_reads=num_reads,
                        num_sweeps=current_num_sweeps,
                        **{k: v for k, v in params.items()
                           if k not in ('num_reads', 'num_sweeps')},
                    )

                    sample_time = time.time() - sample_start
                    self.timing_stats['sampling'].append(
                        sample_time * 1e6,
                    )
                    self.timing_stats['preprocessing'].append(
                        (sample_start - prefilter_start) * 1e6,
                    )
                except Exception as e:
                    if self._on_sampling_error(e, stop_event):
                        self._post_mine_cleanup()
                        return None
                    continue

                sampleset = self._post_sample(sampleset)
                if stop_event.is_set():
                    break

                # Evaluate
                postprocess_start = time.time()
                self.current_stage = 'postprocessing'
                self.current_stage_start = postprocess_start

                self.timing_stats['total_samples'] += len(
                    sampleset.record.energy,
                )
                self.timing_stats['blocks_attempted'] += 1

                result = self.evaluate_sampleset(
                    sampleset, current_requirements, nodes, edges,
                    nonce, salt, prev_timestamp, start_time,
                )

                self.timing_stats['postprocessing'].append(
                    (time.time() - postprocess_start) * 1e6,
                )

                if result:
                    self.logger.info(
                        f"[Block-{cur_index}] Mined! "
                        f"Nonce: {nonce}, Salt: {salt.hex()[:4]}..., "
                        f"Greedy: {greedy_e:.0f}, "
                        f"SA Energy: {result.energy:.2f}, "
                        f"Solutions: {result.num_valid}, "
                        f"Diversity: {result.diversity:.3f}, "
                        f"Time: {time.time() - start_time:.2f}s",
                    )
                    self._post_mine_cleanup()
                    return result

                self.update_top_samples(
                    sampleset, nonce, salt, current_requirements,
                )

            progress += 1
            if progress % 5 == 0:
                best_energy = (
                    min(self.top_attempts[0].sampleset.record.energy)
                    if self.top_attempts
                    else float('inf')
                )
                self.logger.info(
                    f"Progress: {progress} batches "
                    f"({progress * self.PREFILTER_BATCH_SIZE} scored, "
                    f"{progress * self.PREFILTER_KEEP} SA'd), "
                    f"best energy: {best_energy:.2f} | "
                    f"Sweeps: {current_num_sweeps}/{max_num_sweeps}, "
                    f"Reads: {num_reads}",
                )

        self._post_mine_cleanup()
        self.logger.info("Stopping mining, no results found")
        return None
