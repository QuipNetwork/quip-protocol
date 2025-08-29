"""Mining result dataclass and legacy imports for compatibility."""

from shared.base_miner import MiningResult, BaseMiner

# For backward compatibility, re-export key classes
Miner = BaseMiner  # Legacy alias

__all__ = ['MiningResult', 'BaseMiner', 'Miner']