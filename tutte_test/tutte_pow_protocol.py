"""
Tutte Polynomial Proof-of-Work Protocol

Design:
1. Block header determines the proof requirements
2. Miners assemble graphs from rainbow table minors
3. Assembly polynomial must satisfy difficulty threshold

Difficulty is expressed via the Whitney rank polynomial / Tutte polynomial
evaluations at specific points:
  - T(1,1): spanning tree count
  - T(2,1): spanning forest count
  - T(1,2): connected spanning subgraph count

Dynamic adjustment:
  - > 20 blocks/hour → increase difficulty
  - < 5 blocks/hour → decrease difficulty
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutte_test.tutte_utils import TuttePolynomial


class JoinOperation(Enum):
    """How to combine two minors."""
    DISJOINT = "disjoint"      # T(G₁ ∪ G₂) = T(G₁) × T(G₂)
    CUT_VERTEX = "cut_vertex"  # T(G₁ ·₁ G₂) = T(G₁) × T(G₂)


class HardwareArchitecture(Enum):
    """
    Supported quantum annealing hardware architectures.

    Each architecture has characteristic minors that embed efficiently.
    Miners select minors matching their hardware for optimal performance.
    """
    ZEPHYR = "zephyr"      # D-Wave Advantage2 (Z(m,t) topology)
    PEGASUS = "pegasus"    # D-Wave Advantage (P(m) topology)
    GENERIC = "generic"    # Any hardware / classical simulation


@dataclass
class minorReference:
    """Reference to a minor in the rainbow table."""
    name: str
    polynomial: TuttePolynomial = None


@dataclass
class AssemblyStep:
    """One step in assembling a graph from minors."""
    minor: minorReference
    join_op: JoinOperation


@dataclass
class Assembly:
    """Complete assembly recipe from minors."""
    steps: List[AssemblyStep]

    def compute_polynomial(self) -> TuttePolynomial:
        """
        Compute the Tutte polynomial of the assembled graph.

        For disjoint union and cut vertex joins: T = T₁ × T₂
        """
        if not self.steps:
            return TuttePolynomial.one()

        result = self.steps[0].minor.polynomial
        for step in self.steps[1:]:
            result = result * step.minor.polynomial
        return result

    def to_dict(self) -> Dict:
        """Serialize assembly for transmission."""
        return {
            'steps': [
                {'minor': step.minor.name, 'join_op': step.join_op.value}
                for step in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict, rainbow_table: Dict[str, TuttePolynomial]) -> 'Assembly':
        """Deserialize assembly."""
        steps = []
        for step_data in data['steps']:
            name = step_data['minor']
            poly = rainbow_table.get(name)
            steps.append(AssemblyStep(
                minor=minorReference(name=name, polynomial=poly),
                join_op=JoinOperation(step_data['join_op'])
            ))
        return cls(steps=steps)


# =============================================================================
# DIFFICULTY SYSTEM
# =============================================================================

@dataclass
class DifficultyTarget:
    """
    Difficulty specification using Tutte polynomial evaluations.

    Uses the Whitney rank polynomial formulation:
    - T(x,y) encodes graph structure via rank-nullity of edge subsets
    - Specific evaluations give graph invariants:
      T(1,1) = spanning trees
      T(2,1) = spanning forests
      T(1,2) = connected spanning subgraphs
    """
    # Polynomial evaluation thresholds
    min_T_1_1: int = 0      # T(1,1) threshold
    min_T_2_1: int = 0      # T(2,1) threshold
    min_T_2_2: int = 0      # T(2,2) threshold

    # Polynomial structure requirements
    min_x_degree: int = 0   # Rank of graphic matroid ≥ this
    min_y_degree: int = 0   # Nullity bound
    min_terms: int = 0      # Polynomial complexity

    # Difficulty level for tracking
    level: int = 1

    def is_satisfied_by(self, poly: TuttePolynomial) -> bool:
        """Check if a polynomial meets this difficulty."""
        if poly.evaluate(1, 1) < self.min_T_1_1:
            return False
        if poly.evaluate(2, 1) < self.min_T_2_1:
            return False
        if poly.evaluate(2, 2) < self.min_T_2_2:
            return False
        if poly.x_degree() < self.min_x_degree:
            return False
        if poly.y_degree() < self.min_y_degree:
            return False
        if len(poly.coefficients) < self.min_terms:
            return False
        return True

    def scale(self, factor: float) -> 'DifficultyTarget':
        """Scale difficulty by a factor."""
        return DifficultyTarget(
            min_T_1_1=int(self.min_T_1_1 * factor),
            min_T_2_1=int(self.min_T_2_1 * factor),
            min_T_2_2=int(self.min_T_2_2 * factor),
            min_x_degree=self.min_x_degree,
            min_y_degree=self.min_y_degree,
            min_terms=max(self.min_terms, int(self.min_terms * (factor ** 0.3))),
            level=int(self.level * factor)
        )

    def to_dict(self) -> Dict:
        return {
            'min_T_1_1': self.min_T_1_1,
            'min_T_2_1': self.min_T_2_1,
            'min_T_2_2': self.min_T_2_2,
            'min_x_degree': self.min_x_degree,
            'min_y_degree': self.min_y_degree,
            'min_terms': self.min_terms,
            'level': self.level
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DifficultyTarget':
        return cls(**data)

    @classmethod
    def from_polynomial(cls, poly: TuttePolynomial, level: int = 1) -> 'DifficultyTarget':
        """Create difficulty target from a reference polynomial."""
        return cls(
            min_T_1_1=poly.evaluate(1, 1),
            min_T_2_1=poly.evaluate(2, 1),
            min_T_2_2=poly.evaluate(2, 2),
            min_x_degree=poly.x_degree(),
            min_y_degree=poly.y_degree(),
            min_terms=len(poly.coefficients),
            level=level
        )


def load_baseline_from_rainbow_table(table_path: str, baseline_name: str = 'Z(1,1)') -> DifficultyTarget:
    """
    Load baseline difficulty from a polynomial in the rainbow table.

    Uses an actual computed polynomial as the reference, not made-up values.
    """
    with open(table_path) as f:
        data = json.load(f)

    for key, entry in data.get('graphs', {}).items():
        if entry.get('name') == baseline_name:
            # Reconstruct polynomial
            coeffs = {}
            for k, v in entry.get('coefficients', {}).items():
                i, j = map(int, k.split(','))
                coeffs[(i, j)] = v
            poly = TuttePolynomial.from_dict(coeffs)
            return DifficultyTarget.from_polynomial(poly)

    raise ValueError(f"Baseline '{baseline_name}' not found in rainbow table")


@dataclass
class DifficultyAdjuster:
    """
    Dynamic difficulty adjustment based on block rate.

    Target: 10-15 blocks per hour
    - > 20 blocks/hour → increase difficulty by 25%
    - < 5 blocks/hour → decrease difficulty by 20%
    """
    min_blocks_per_hour: int = 5
    max_blocks_per_hour: int = 20
    increase_factor: float = 1.25
    decrease_factor: float = 0.80

    block_timestamps: List[float] = field(default_factory=list)
    current_difficulty: DifficultyTarget = None

    def __post_init__(self):
        if self.current_difficulty is None:
            # Load baseline from rainbow table
            table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
            try:
                self.current_difficulty = load_baseline_from_rainbow_table(table_path)
            except (FileNotFoundError, ValueError):
                # Fallback minimal difficulty
                self.current_difficulty = DifficultyTarget(
                    min_T_1_1=1000,
                    min_x_degree=5,
                    min_terms=10,
                    level=1
                )

    def record_block(self, timestamp: float = None):
        """Record a new block being mined."""
        if timestamp is None:
            timestamp = time.time()
        self.block_timestamps.append(timestamp)

        # Keep only blocks from last hour
        one_hour_ago = timestamp - 3600
        self.block_timestamps = [t for t in self.block_timestamps if t > one_hour_ago]

    def get_blocks_per_hour(self) -> int:
        """Calculate current block rate."""
        if len(self.block_timestamps) < 2:
            return 12  # Default if not enough data

        now = time.time()
        one_hour_ago = now - 3600
        return len([t for t in self.block_timestamps if t > one_hour_ago])

    def adjust_difficulty(self) -> DifficultyTarget:
        """Adjust difficulty based on recent block rate."""
        blocks_per_hour = self.get_blocks_per_hour()

        if blocks_per_hour > self.max_blocks_per_hour:
            self.current_difficulty = self.current_difficulty.scale(self.increase_factor)
        elif blocks_per_hour < self.min_blocks_per_hour:
            self.current_difficulty = self.current_difficulty.scale(self.decrease_factor)

        return self.current_difficulty

    def get_current_difficulty(self) -> DifficultyTarget:
        """Get current difficulty without adjustment."""
        return self.current_difficulty


# =============================================================================
# PROOF OF WORK
# =============================================================================

@dataclass
class ProofOfWork:
    """A proof-of-work solution."""
    block_header: bytes
    nonce: int
    assembly: Assembly
    timestamp: float = field(default_factory=time.time)
    result_polynomial: TuttePolynomial = None

    def __post_init__(self):
        if self.result_polynomial is None:
            self.result_polynomial = self.assembly.compute_polynomial()

    def verify(self, difficulty: DifficultyTarget, rainbow_table: Dict[str, TuttePolynomial]) -> bool:
        """Verify this proof-of-work."""
        # Verify all minors exist in rainbow table
        for step in self.assembly.steps:
            if step.minor.name not in rainbow_table:
                return False
            if step.minor.polynomial != rainbow_table[step.minor.name]:
                return False

        # Recompute polynomial (don't trust claimed result)
        computed = self.assembly.compute_polynomial()

        # Check difficulty
        return difficulty.is_satisfied_by(computed)

    def to_dict(self) -> Dict:
        return {
            'block_header': self.block_header.hex(),
            'nonce': self.nonce,
            'assembly': self.assembly.to_dict(),
            'timestamp': self.timestamp,
            'T_1_1': self.result_polynomial.evaluate(1, 1),
            'T_2_1': self.result_polynomial.evaluate(2, 1),
        }


# =============================================================================
# MINER
# =============================================================================

class TuttePowMiner:
    """
    Miner that finds valid proof-of-work solutions.

    Supports hardware-specific mining strategies:
    - ZEPHYR: Prefers Z(1,1) and Zephyr_minor_* entries
    - PEGASUS: Prefers Pegasus_minor_* entries
    - GENERIC: Uses all available minors
    """

    def __init__(self, rainbow_table_path: str = None,
                 hardware: HardwareArchitecture = HardwareArchitecture.GENERIC):
        """
        Load rainbow table of minors.

        Args:
            rainbow_table_path: Path to rainbow table JSON
            hardware: Hardware architecture for minor filtering
        """
        if rainbow_table_path is None:
            rainbow_table_path = os.path.join(
                os.path.dirname(__file__), 'tutte_rainbow_table.json'
            )

        self.hardware = hardware

        with open(rainbow_table_path) as f:
            data = json.load(f)

        # Load all minors
        self.all_minors: Dict[str, TuttePolynomial] = {}
        for key, entry in data.get('graphs', {}).items():
            name = entry.get('name', key)
            coeffs = {}
            for k, v in entry.get('coefficients', {}).items():
                i, j = map(int, k.split(','))
                coeffs[(i, j)] = v
            self.all_minors[name] = TuttePolynomial.from_dict(coeffs)

        # Filter minors based on hardware architecture
        self.minors = self._filter_minors_for_hardware()

    def _filter_minors_for_hardware(self) -> Dict[str, TuttePolynomial]:
        """
        Filter minors that embed efficiently on the target hardware.

        Hardware-specific minors are prioritized, but standard graph
        families (K_n, grids, etc.) are included as they often embed well.
        """
        if self.hardware == HardwareArchitecture.GENERIC:
            return self.all_minors.copy()

        filtered = {}

        for name, poly in self.all_minors.items():
            # Always include hardware-specific minors
            if self.hardware == HardwareArchitecture.ZEPHYR:
                if 'Zephyr' in name or name == 'Z(1,1)':
                    filtered[name] = poly
                    continue
            elif self.hardware == HardwareArchitecture.PEGASUS:
                if 'Pegasus' in name:
                    filtered[name] = poly
                    continue

            # Include standard graph families (good embeddings on most hardware)
            if any(prefix in name for prefix in ['K_', 'C_', 'P_', 'Grid_',
                                                   'Ladder_', 'W_', 'Q_', 'Circ_']):
                filtered[name] = poly

        return filtered

    def get_hardware_summary(self) -> str:
        """Return summary of available minors for this hardware."""
        zephyr_count = sum(1 for n in self.minors if 'Zephyr' in n or n == 'Z(1,1)')
        pegasus_count = sum(1 for n in self.minors if 'Pegasus' in n)
        standard_count = len(self.minors) - zephyr_count - pegasus_count

        return (f"Hardware: {self.hardware.value}, "
                f"minors: {len(self.minors)} total "
                f"({zephyr_count} Zephyr, {pegasus_count} Pegasus, {standard_count} standard)")

    def find_assembly(self, difficulty: DifficultyTarget) -> Optional[Assembly]:
        """Find an assembly of minors that satisfies difficulty."""
        # Sort minors by T(1,1) evaluation
        sorted_minors = sorted(
            self.minors.items(),
            key=lambda x: x[1].evaluate(1, 1),
            reverse=True
        )

        # Try single minors
        for name, poly in sorted_minors:
            if difficulty.is_satisfied_by(poly):
                step = AssemblyStep(
                    minor=minorReference(name=name, polynomial=poly),
                    join_op=JoinOperation.DISJOINT
                )
                return Assembly(steps=[step])

        # Try pairs (multiplicative composition)
        for name1, poly1 in sorted_minors[:30]:
            for name2, poly2 in sorted_minors[:30]:
                combined = poly1 * poly2
                if difficulty.is_satisfied_by(combined):
                    return Assembly(steps=[
                        AssemblyStep(minorReference(name1, poly1), JoinOperation.DISJOINT),
                        AssemblyStep(minorReference(name2, poly2), JoinOperation.CUT_VERTEX),
                    ])

        # Try triples
        for name1, poly1 in sorted_minors[:15]:
            for name2, poly2 in sorted_minors[:15]:
                for name3, poly3 in sorted_minors[:15]:
                    combined = poly1 * poly2 * poly3
                    if difficulty.is_satisfied_by(combined):
                        return Assembly(steps=[
                            AssemblyStep(minorReference(name1, poly1), JoinOperation.DISJOINT),
                            AssemblyStep(minorReference(name2, poly2), JoinOperation.CUT_VERTEX),
                            AssemblyStep(minorReference(name3, poly3), JoinOperation.CUT_VERTEX),
                        ])

        return None

    def mine(self, block_header: bytes, difficulty: DifficultyTarget) -> Optional[ProofOfWork]:
        """Mine for a valid proof-of-work."""
        assembly = self.find_assembly(difficulty)
        if assembly:
            return ProofOfWork(
                block_header=block_header,
                nonce=0,
                assembly=assembly
            )
        return None


# =============================================================================
# DEMO
# =============================================================================

def demo_protocol():
    """Demonstrate the Tutte PoW protocol with multiple hardware architectures."""
    print("=" * 70)
    print("TUTTE POLYNOMIAL PROOF-OF-WORK PROTOCOL")
    print("=" * 70)

    # Initialize miners for different hardware
    try:
        generic_miner = TuttePowMiner(hardware=HardwareArchitecture.GENERIC)
        zephyr_miner = TuttePowMiner(hardware=HardwareArchitecture.ZEPHYR)
        pegasus_miner = TuttePowMiner(hardware=HardwareArchitecture.PEGASUS)
    except FileNotFoundError:
        print("Rainbow table not found. Run rainbow_table.py first.")
        return

    print("\n--- Hardware Configurations ---")
    print(f"  {generic_miner.get_hardware_summary()}")
    print(f"  {zephyr_miner.get_hardware_summary()}")
    print(f"  {pegasus_miner.get_hardware_summary()}")

    # Initialize difficulty from rainbow table baseline
    adjuster = DifficultyAdjuster()
    baseline = adjuster.get_current_difficulty()

    print(f"\n--- Baseline Difficulty (from Z(1,1) in rainbow table) ---")
    print(f"  T(1,1) ≥ {baseline.min_T_1_1:,}")
    print(f"  T(2,1) ≥ {baseline.min_T_2_1:,}")
    print(f"  T(2,2) ≥ {baseline.min_T_2_2:,}")
    print(f"  x-degree ≥ {baseline.min_x_degree}")
    print(f"  y-degree ≥ {baseline.min_y_degree}")
    print(f"  terms ≥ {baseline.min_terms}")

    # Demonstrate mining with different hardware
    print(f"\n--- Mining with Different Hardware Architectures ---")
    block_header = b"genesis"
    difficulty = adjuster.get_current_difficulty()

    for name, miner in [("GENERIC", generic_miner),
                        ("ZEPHYR", zephyr_miner),
                        ("PEGASUS", pegasus_miner)]:
        print(f"\n{name} Miner:")
        proof = miner.mine(block_header, difficulty)

        if proof:
            minor_names = [s.minor.name for s in proof.assembly.steps]
            print(f"  Assembly: {minor_names}")
            print(f"  T(1,1) = {proof.result_polynomial.evaluate(1, 1):,}")
            print(f"  T(2,1) = {proof.result_polynomial.evaluate(2, 1):,}")
        else:
            print(f"  No valid assembly found with available minors")

    # Demonstrate difficulty adjustment
    print(f"\n--- Difficulty Adjustment ---")

    print("\nScenario: 25 blocks/hour (too fast)")
    fast_adjuster = DifficultyAdjuster()
    now = time.time()
    for i in range(25):
        fast_adjuster.record_block(now - i * 120)
    new_diff = fast_adjuster.adjust_difficulty()
    print(f"  T(1,1) threshold: {fast_adjuster.current_difficulty.min_T_1_1:,} → {new_diff.min_T_1_1:,}")

    print("\nScenario: 3 blocks/hour (too slow)")
    slow_adjuster = DifficultyAdjuster()
    for i in range(3):
        slow_adjuster.record_block(now - i * 1000)
    new_diff = slow_adjuster.adjust_difficulty()
    print(f"  T(1,1) threshold: {slow_adjuster.current_difficulty.min_T_1_1:,} → {new_diff.min_T_1_1:,}")

    print("\n" + "=" * 70)
    print("PROTOCOL SUMMARY")
    print("=" * 70)
    print("""
HARDWARE SUPPORT:
  • ZEPHYR: D-Wave Advantage2 topology (Z(m,t) graphs)
    - Uses Z(1,1) and Zephyr minors for efficient embedding
  • PEGASUS: D-Wave Advantage topology (P(m) graphs)
    - Uses Pegasus minors extracted from P(2)
  • GENERIC: Any hardware / classical simulation
    - Uses all available minors

DIFFICULTY:
  Based on Tutte polynomial evaluations (Whitney rank polynomial):
  • T(1,1): spanning tree count
  • T(2,1): spanning forest count
  • T(2,2): general evaluation

  Adjusted dynamically:
  • > 20 blocks/hour → difficulty × 1.25
  • < 5 blocks/hour  → difficulty × 0.80

MINING:
  1. Get current difficulty from network
  2. Select minors matching hardware architecture
  3. Combine via polynomial multiplication (1-sum)
  4. Verify T(assembly) satisfies all thresholds

VERIFICATION:
  1. Check minors exist in rainbow table
  2. Verify polynomial multiplication
  3. Confirm evaluations meet difficulty
  4. O(k) verification for k minors
""")


if __name__ == "__main__":
    demo_protocol()
