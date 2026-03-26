"""QPU mining components for quantum blockchain.

Supports multiple quantum hardware backends:
  - DWave: Quantum annealing via D-Wave hardware
  - IBM: Gate-based QAOA via IBM Quantum / Aer simulator (requires qiskit)
"""

try:
    from QPU.DWave import DWaveMiner, DWaveSamplerWrapper
    from QPU.DWave import QPUTimeManager, QPUTimeConfig, QPUTimeEstimate, parse_duration
    _HAS_DWAVE = True
except ImportError:
    _HAS_DWAVE = False

try:
    from QPU.IBM import IBMQAOAMiner, QAOASolverWrapper
    _HAS_IBM = True
except ImportError:
    _HAS_IBM = False

__all__ = []

if _HAS_DWAVE:
    __all__ += [
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