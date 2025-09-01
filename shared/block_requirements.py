"""Block requirements data structure for quantum blockchain."""

import struct
from dataclasses import dataclass
from typing import Dict


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


# For backward compatibility, create an alias
NextBlockRequirements = BlockRequirements
