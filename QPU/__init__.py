"""QPU mining components for quantum blockchain.

Supports multiple quantum hardware backends:
  - DWave: Quantum annealing via D-Wave hardware
  - IBM: Gate-based QAOA via IBM Quantum / Aer simulator (requires qiskit)
"""

import logging

logger = logging.getLogger(__name__)

try:
    from QPU.DWave import DWaveMiner, DWaveSamplerWrapper
    from QPU.DWave import QPUTimeManager, QPUTimeConfig, QPUTimeEstimate, parse_duration
    _HAS_DWAVE = True
except ImportError as e:
    # Logged so a broken internal import isn't silently mistaken for an
    # uninstalled D-Wave SDK. A missing optional dependency is expected;
    # an error mentioning a QPU/shared module is a real bug worth seeing.
    logger.warning("D-Wave QPU backend unavailable: %s", e)
    _HAS_DWAVE = False

try:
    from QPU.IBM import IBMQAOAMiner, QAOASolverWrapper
    _HAS_IBM = True
except ImportError as e:
    # qiskit/qiskit-aer are optional and not installed by default, so a missing
    # qiskit module is expected (debug). Any other ImportError points at a real
    # bug inside QPU.IBM and should be visible (warning) rather than silenced.
    if e.name and e.name.startswith("qiskit"):
        logger.debug("IBM QAOA backend unavailable (qiskit not installed): %s", e)
    else:
        logger.warning("IBM QAOA backend import failed unexpectedly: %s", e)
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