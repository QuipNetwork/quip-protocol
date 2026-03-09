"""QPU mining components for quantum blockchain.

Supports multiple quantum hardware backends:
  - DWave: Quantum annealing via D-Wave hardware
  - IBM: Gate-based QAOA via IBM Quantum / Aer simulator (requires qiskit)
"""

from .DWave import DWaveMiner, DWaveSamplerWrapper
from .DWave import QPUTimeManager, QPUTimeConfig, QPUTimeEstimate, parse_duration

try:
    from .IBM import IBMQAOAMiner, QAOASolverWrapper
    _HAS_IBM = True
except ImportError:
    _HAS_IBM = False

__all__ = [
    # D-Wave
    'DWaveMiner',
    'DWaveSamplerWrapper',
    'QPUTimeManager',
    'QPUTimeConfig',
    'QPUTimeEstimate',
    'parse_duration',
]

if _HAS_IBM:
    __all__ += [
        'IBMQAOAMiner',
        'QAOASolverWrapper',
    ]
