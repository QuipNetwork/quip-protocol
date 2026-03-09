"""QPU mining components for quantum blockchain.

Supports multiple quantum hardware backends:
  - DWave: Quantum annealing via D-Wave hardware
  - IBM: Gate-based QAOA via IBM Quantum / Aer simulator
"""

from .DWave import DWaveMiner, DWaveSamplerWrapper
from .IBM import IBMQAOAMiner, QAOASolverWrapper
from .DWave import QPUTimeManager, QPUTimeConfig, QPUTimeEstimate, parse_duration

__all__ = [
    # D-Wave
    'DWaveMiner',
    'DWaveSamplerWrapper',
    'QPUTimeManager',
    'QPUTimeConfig',
    'QPUTimeEstimate',
    'parse_duration',
    # IBM
    'IBMQAOAMiner',
    'QAOASolverWrapper',
]
