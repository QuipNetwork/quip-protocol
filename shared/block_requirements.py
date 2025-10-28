"""Block requirements data structure for quantum blockchain."""

from asyncio.log import logger
import logging
import random
from typing import Optional

from shared.block import Block, BlockRequirements
from shared.energy_utils import adjust_energy_along_curve
from shared.miner_types import MiningResult
from shared.quantum_proof_of_work import validate_quantum_proof
from shared.time_utils import utc_timestamp, validate_block_timestamp

internal_logger = logging.getLogger(__name__)

##
# Block Requirements adjustments for QPoW
##


def calculate_adjustment_rate_with_randomness(
    mining_time: float,
    direction: str
) -> float:
    """Calculate time-based randomized adjustment rate for difficulty changes.

    Uses mining time to determine base adjustment rate and variance, with
    linear interpolation for intermediate times.

    Args:
        mining_time: Time taken to mine the block in seconds
        direction: Either 'harder' or 'easier'

    Returns:
        Randomized adjustment rate (e.g., 0.35 for 35%)

    For HARDENING ('harder'):
        - < 360s: 35% ± 30% (range: 5% to 65%)
        - > 600s: 5% ± 4% (range: 1% to 9%)
        - 360s-600s: Linear interpolation

    For EASING ('easier'):
        - > 1200s: 15% ± 14% (range: 1% to 29%)
        - < 600s: 2.5% ± 2% (range: 0.5% to 4.5%)
        - 600s-1200s: Linear interpolation
    """
    if direction == 'harder':
        # HARDENING ranges
        if mining_time < 360.0:
            base_rate = 0.35
            variance = 0.30
        elif mining_time > 600.0:
            base_rate = 0.05
            variance = 0.04
        else:
            # Linear interpolation between 360s and 600s
            # At 360s: rate=35%, variance=30%
            # At 600s: rate=5%, variance=4%
            progress = (mining_time - 360.0) / (600.0 - 360.0)
            base_rate = 0.35 - progress * (0.35 - 0.05)
            variance = 0.30 - progress * (0.30 - 0.04)

    else:  # 'easier'
        # EASING ranges
        if mining_time > 1200.0:
            base_rate = 0.15
            variance = 0.14
        elif mining_time < 600.0:
            base_rate = 0.025
            variance = 0.02
        else:
            # Linear interpolation between 600s and 1200s
            # At 600s: rate=2.5%, variance=2%
            # At 1200s: rate=15%, variance=14%
            progress = (mining_time - 600.0) / (1200.0 - 600.0)
            base_rate = 0.025 + progress * (0.15 - 0.025)
            variance = 0.02 + progress * (0.14 - 0.02)

    # Apply randomness within variance
    min_rate = max(0.001, base_rate - variance)  # Ensure at least 0.1%
    max_rate = base_rate + variance
    adjustment_rate = random.uniform(min_rate, max_rate)

    return adjustment_rate


def compute_current_requirements(
    initial_requirements: BlockRequirements,
    prev_timestamp: int,
    log: logging.Logger = internal_logger,
    current_time: Optional[int] = None
) -> BlockRequirements:
    """
    Compute current block requirements with timeout-based difficulty decay applied.

    Args:
        initial_requirements: The original block requirements
        prev_timestamp: Timestamp of the previous block
        logger: Optional logger for recording decay changes

    Returns:
        BlockRequirements with decay applied if elapsed time warrants it
    """
    if current_time is None:
        current_time = utc_timestamp()

    if initial_requirements.timeout_to_difficulty_adjustment_decay <= 0:
        return initial_requirements

    elapsed = max(0, int((current_time - prev_timestamp) / initial_requirements.timeout_to_difficulty_adjustment_decay))


    if elapsed == 0:
        return initial_requirements

    log.debug(f"Elapsed time: {elapsed} steps ({current_time - prev_timestamp}s, {initial_requirements.timeout_to_difficulty_adjustment_decay}s per step)")

    # Apply decay for each elapsed step
    req_dict = initial_requirements.to_json()
    for _ in range(elapsed):
        req_dict = calculate_requirements_decay(req_dict)

    decayed_requirements = BlockRequirements.from_json(req_dict)

    # Log changes only if decay was applied
    if elapsed > 0:
        log.info(
            f"Applied {elapsed} difficulty decay steps: "
            f"energy {initial_requirements.difficulty_energy:.2f} -> {decayed_requirements.difficulty_energy:.2f}, "
            f"diversity {initial_requirements.min_diversity:.3f} -> {decayed_requirements.min_diversity:.3f}, "
            f"solutions {initial_requirements.min_solutions} -> {decayed_requirements.min_solutions}"
        )

    return decayed_requirements

def calculate_requirements_decay(cur_requirements: dict) -> dict:
    """
    Apply one step of timeout-based difficulty decay to the given requirements.

    Expects a dict-like with keys:
      - difficulty_energy (float, typically negative)
      - min_diversity (float)
      - min_solutions (int)
      - timeout_to_difficulty_adjustment_decay (int seconds)

    Returns a new dict with eased (less strict) requirements.

    Notes:
    - Uses curve-based energy adjustment at half the rate of difficulty increases
    - Energies are negative; easing moves the threshold closer to 0.
    - Diversity and min_solutions ease downward within blockchain-defined range constraints
    - Minimum energy adjustment is 3 (vs 5 for difficulty adjustments).
    """
    from shared.energy_utils import DEFAULT_DIVERSITY_RANGE, DEFAULT_SOLUTIONS_RANGE

    # Base easing rates (half the rate of difficulty adjustments)
    energy_ease_rate = 0.025      # 2.5% easier per decay step (half of 5%)
    diversity_ease_rate = 0.01    # 1% easier per decay step (half of 2%)
    solutions_ease_rate = 0.05    # 5% easier per decay step (half of 10%)

    # Use blockchain-defined ranges for floors/ceilings
    # Currently fixed at (0.3, 0.3) and (20, 20) but infrastructure ready for chain consensus
    MIN_DIVERSITY_FLOOR = DEFAULT_DIVERSITY_RANGE[0]
    MAX_DIVERSITY_CEILING = DEFAULT_DIVERSITY_RANGE[1]
    MIN_SOLUTIONS_FLOOR = DEFAULT_SOLUTIONS_RANGE[0]
    MAX_SOLUTIONS_CEILING = DEFAULT_SOLUTIONS_RANGE[1]

    de = float(cur_requirements.get('difficulty_energy', 0.0))
    md = float(cur_requirements.get('min_diversity', 0.0))
    ms = int(cur_requirements.get('min_solutions', 0))
    decay = int(cur_requirements.get('timeout_to_difficulty_adjustment_decay', 30))

    # Apply curve-based easing for energy (move toward easier/less negative)
    curve_energy = adjust_energy_along_curve(de, energy_ease_rate, 'easier')

    # Apply minimum adjustment of 3 units for decay
    energy_delta = curve_energy - de
    min_adjustment = 3.0
    if abs(energy_delta) > 0 and abs(energy_delta) < min_adjustment:
        new_de = de + min_adjustment  # Always easier for decay
    else:
        new_de = curve_energy

    # Ease diversity and solutions downward within blockchain-defined limits
    new_md = min(MAX_DIVERSITY_CEILING, max(MIN_DIVERSITY_FLOOR, md - diversity_ease_rate))
    new_ms = min(MAX_SOLUTIONS_CEILING, max(MIN_SOLUTIONS_FLOOR, int(ms * (1 - solutions_ease_rate))))

    return {
        'difficulty_energy': float(new_de),
        'min_diversity': float(new_md),
        'min_solutions': int(new_ms),
        'timeout_to_difficulty_adjustment_decay': decay,
    }


def compute_next_block_requirements(previous_block: Block, mining_result: MiningResult,
                                    log: logging.Logger = logger) -> BlockRequirements:
    """
    Compute the next block requirements based on the previous block and mining result.

    Rules:
    - Always HARDEN difficulty if the last block was mined in under 60 seconds
    - Otherwise:
        - If the same miner type wins consecutively, EASE difficulty
        - If a different miner type wins, HARDEN difficulty

    Uses curve-based energy adjustments instead of flat multiplication.
    Energy curve: min (-16000) to max (-14000) with knee at (-15600).

    Note: min_diversity and min_solutions are constrained by blockchain-defined ranges
    (currently fixed at 0.3 and 20, but infrastructure ready for chain consensus).
    """
    from shared.energy_utils import DEFAULT_DIVERSITY_RANGE, DEFAULT_SOLUTIONS_RANGE

    # Get current requirements from previous block
    prev_req = previous_block.next_block_requirements
    if not prev_req:
        raise ValueError("Previous block has no next block requirements")

    if previous_block.header.index > 0:
        prev_req = compute_current_requirements(prev_req, previous_block.header.timestamp, log, mining_result.timestamp)

    # Extract miner type from mining result
    current_winner = mining_result.miner_id

    # Get the previous winner from the previous block's miner info
    prev_winner = None
    if previous_block.miner_info:
        prev_miner_id = previous_block.miner_info.miner_id
        prev_winner = prev_miner_id.split('-')[1] if '-' in prev_miner_id else prev_miner_id

    # Get mining time for time-based adjustment calculation
    mining_time = mining_result.mining_time if mining_result.mining_time is not None else 600.0

    # Use blockchain-defined ranges for constraints
    # Currently fixed at (0.3, 0.3) and (20, 20) but infrastructure ready for chain consensus
    MIN_DIVERSITY = DEFAULT_DIVERSITY_RANGE[0]
    MAX_DIVERSITY = DEFAULT_DIVERSITY_RANGE[1]
    MIN_SOLUTIONS = DEFAULT_SOLUTIONS_RANGE[0]
    MAX_SOLUTIONS = DEFAULT_SOLUTIONS_RANGE[1]

    # Helper function to apply minimum adjustment for difficulty changes
    def apply_min_adjustment(old_energy: float, new_energy: float, direction: str, min_adj: float = 5.0) -> float:
        energy_delta = new_energy - old_energy
        if abs(energy_delta) > 0 and abs(energy_delta) < min_adj:
            if direction == 'harder':
                return old_energy - min_adj
            else:  # 'easier'
                return old_energy + min_adj
        return new_energy

    # If block was mined too quickly, always HARDEN
    if mining_result.mining_time is not None and mining_result.mining_time < 360.0:
        # Calculate time-based randomized adjustment rate
        energy_adjustment_rate = calculate_adjustment_rate_with_randomness(mining_time, 'harder')

        curve_energy = adjust_energy_along_curve(prev_req.difficulty_energy, energy_adjustment_rate, 'harder')
        new_difficulty_energy = apply_min_adjustment(prev_req.difficulty_energy, curve_energy, 'harder')
        new_min_diversity = min(MAX_DIVERSITY, prev_req.min_diversity + energy_adjustment_rate * 0.4)  # Scale for diversity
        new_min_solutions = min(MAX_SOLUTIONS, int(prev_req.min_solutions * (1 + energy_adjustment_rate * 2)))  # Scale for solutions

        log.info(
            f"Block was mined in {mining_result.mining_time:.2f}s (<360s) - HARDENING difficulty (rate: {energy_adjustment_rate:.1%})")
        log.info(f"  Energy: {prev_req.difficulty_energy:.1f} -> {new_difficulty_energy:.1f}")
        log.info(f"  Diversity: {prev_req.min_diversity:.2f} -> {new_min_diversity:.2f}")
        log.info(f"  Solutions: {prev_req.min_solutions} -> {new_min_solutions}")
    else:
        log.info(f"Last winner: {prev_winner}, current winner: {current_winner}")
        if current_winner == prev_winner:
            # Same miner won again - make it EASIER
            # Calculate time-based randomized adjustment rate
            energy_adjustment_rate = calculate_adjustment_rate_with_randomness(mining_time, 'easier')

            # Higher energy threshold (less negative), lower diversity/solutions
            curve_energy = adjust_energy_along_curve(prev_req.difficulty_energy, energy_adjustment_rate, 'easier')
            new_difficulty_energy = apply_min_adjustment(prev_req.difficulty_energy, curve_energy, 'easier')
            new_min_diversity = max(MIN_DIVERSITY, prev_req.min_diversity - energy_adjustment_rate * 0.4)  # Scale for diversity
            new_min_solutions = max(MIN_SOLUTIONS, int(prev_req.min_solutions * (1 - energy_adjustment_rate * 2)))  # Scale for solutions

            log.info(f"Same miner type ({current_winner}) won - EASING difficulty (rate: {energy_adjustment_rate:.1%}, time: {mining_time:.1f}s)")
            log.info(f"  Energy: {prev_req.difficulty_energy:.1f} -> {new_difficulty_energy:.1f}")
            log.info(f"  Diversity: {prev_req.min_diversity:.2f} -> {new_min_diversity:.2f}")
            log.info(f"  Solutions: {prev_req.min_solutions} -> {new_min_solutions}")
        else:
            # Different miner won - make it HARDER
            # Calculate time-based randomized adjustment rate
            energy_adjustment_rate = calculate_adjustment_rate_with_randomness(mining_time, 'harder')

            # Lower energy threshold (more negative), higher diversity/solutions
            curve_energy = adjust_energy_along_curve(prev_req.difficulty_energy, energy_adjustment_rate, 'harder')
            new_difficulty_energy = apply_min_adjustment(prev_req.difficulty_energy, curve_energy, 'harder')
            new_min_diversity = min(MAX_DIVERSITY, prev_req.min_diversity + energy_adjustment_rate * 0.4)  # Scale for diversity
            new_min_solutions = min(MAX_SOLUTIONS, int(prev_req.min_solutions * (1 + energy_adjustment_rate * 2)))  # Scale for solutions

            log.info(f"Different miner type won ({prev_winner} -> {current_winner}) - HARDENING difficulty (rate: {energy_adjustment_rate:.1%}, time: {mining_time:.1f}s)")
            log.info(f"  Energy: {prev_req.difficulty_energy:.1f} -> {new_difficulty_energy:.1f}")
            log.info(f"  Diversity: {prev_req.min_diversity:.2f} -> {new_min_diversity:.2f}")
            log.info(f"  Solutions: {prev_req.min_solutions} -> {new_min_solutions}")

    return BlockRequirements(
        difficulty_energy=new_difficulty_energy,
        min_diversity=new_min_diversity,
        min_solutions=new_min_solutions,
        timeout_to_difficulty_adjustment_decay=prev_req.timeout_to_difficulty_adjustment_decay
    )

def validate_block(block: 'Block', previous_block: 'Block', logger: logging.Logger = internal_logger) -> bool:
    """Validate this block against the previous block requirements.

    This method validates the quantum proof and other block artifacts.

    The signature is not checked at this time, but it could be as 
    all blocks have miner info, although checking signature is responsibility of 
    the network node layer. 

    Args:
        previous_block: The previous block containing requirements

    Returns:
        True if block is valid, False otherwise
    """
    if not block.quantum_proof or not block.miner_info:
        logger.error(f"Block {block.header.index} rejected: missing quantum proof or miner info")
        return False

    # Get requirements from previous block
    requirements = previous_block.next_block_requirements
    if not requirements:
        logger.error(f"Block {block.header.index} rejected: missing next block requirements")
        return False
    
    # Apply timeout-based difficulty decay based on elapsed time since previous block
    if previous_block.header.index > 0:
        requirements = compute_current_requirements(requirements, previous_block.header.timestamp, logger, block.header.timestamp)

    #Validate the timestamps in the block using UTC time
    if not validate_block_timestamp(block.header.timestamp, previous_block.header.timestamp):
        logger.info(f"Block {block.header.index} rejected: invalid timestamp {block.header.timestamp}")
        return False
    min_gap = block.header.timestamp - (block.header.timestamp - int(block.quantum_proof.mining_time))
    if (block.header.timestamp - min_gap) < previous_block.header.timestamp:
        logger.info(f"Block {block.header.index} rejected: timestamp {block.header.timestamp} - min_gap {min_gap} < previous block timestamp {previous_block.header.timestamp}")
        return False

    # Validate quantum proof against (possibly decayed) requirements
    return validate_quantum_proof(
        block.quantum_proof, 
        block.miner_info.miner_id, 
        requirements, 
        block.header.index, 
        block.header.previous_hash
    )