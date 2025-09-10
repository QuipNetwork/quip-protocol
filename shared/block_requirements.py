"""Block requirements data structure for quantum blockchain."""

import logging
import struct
import time
from dataclasses import dataclass
from typing import Optional

from shared.energy_utils import adjust_energy_along_curve

internal_logger = logging.getLogger(__name__)

@dataclass
class BlockRequirements:
    """Requirements that the next block must satisfy."""
    difficulty_energy: float
    min_diversity: float
    min_solutions: int
    timeout_to_difficulty_adjustment_decay: int

    def to_network(self) -> bytes:
        """Serialize to binary format."""
        result = b''
        result += struct.pack('!d', self.difficulty_energy)
        result += struct.pack('!d', self.min_diversity)
        result += struct.pack('!I', self.min_solutions)
        result += struct.pack('!i', self.timeout_to_difficulty_adjustment_decay)
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'BlockRequirements':
        """Deserialize from binary format."""
        offset = 0
        difficulty_energy = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8
        min_diversity = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8
        min_solutions = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        timeout_to_difficulty_adjustment_decay = struct.unpack('!i', data[offset:offset+4])[0]

        return cls(
            difficulty_energy=difficulty_energy,
            min_diversity=min_diversity,
            min_solutions=min_solutions,
            timeout_to_difficulty_adjustment_decay=timeout_to_difficulty_adjustment_decay
        )

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            'difficulty_energy': self.difficulty_energy,
            'min_diversity': self.min_diversity,
            'min_solutions': self.min_solutions,
            'timeout_to_difficulty_adjustment_decay': self.timeout_to_difficulty_adjustment_decay
        }

    @classmethod
    def from_json(cls, data: dict) -> 'BlockRequirements':
        """Deserialize from JSON-compatible dictionary."""
        return cls(
            difficulty_energy=float(data['difficulty_energy']),
            min_diversity=float(data['min_diversity']),
            min_solutions=int(data['min_solutions']),
            timeout_to_difficulty_adjustment_decay=int(data['timeout_to_difficulty_adjustment_decay'])
        )


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
        current_time = int(time.time())

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
    - Diversity and min_solutions also ease downward within sensible floors.
    - Minimum energy adjustment is 3 (vs 5 for difficulty adjustments).
    """
    # Base easing rates (half the rate of difficulty adjustments)
    energy_ease_rate = 0.025      # 2.5% easier per decay step (half of 5%)
    diversity_ease_rate = 0.01    # 1% easier per decay step (half of 2%)
    solutions_ease_rate = 0.05    # 5% easier per decay step (half of 10%)

    # Floors to avoid collapsing difficulty entirely
    MIN_DIVERSITY_FLOOR = 0.20
    MIN_SOLUTIONS_FLOOR = 10

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

    # Ease diversity and solutions downward within floors
    new_md = max(MIN_DIVERSITY_FLOOR, md - diversity_ease_rate)
    new_ms = max(MIN_SOLUTIONS_FLOOR, int(ms * (1 - solutions_ease_rate)))

    return {
        'difficulty_energy': float(new_de),
        'min_diversity': float(new_md),
        'min_solutions': int(new_ms),
        'timeout_to_difficulty_adjustment_decay': decay,
    }

