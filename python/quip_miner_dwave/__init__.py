"""quip-miner-dwave: D-Wave quantum-annealing miner for the v0.3 protocol.

Binary entry point: ``quip-dwave-qa`` (``python -m quip_miner_dwave``).
"""

__version__ = "0.3.0"

# sysexits-style exit codes (mirrored by quip_protocol::session::ExitCode)
EXIT_CLEAN = 0
EXIT_CONFIG_INVALID = 64
EXIT_ENV_INCOMPATIBLE = 69
EXIT_INTERNAL_FATAL = 70
EXIT_TOKEN_REJECTED = 77

BACKEND = "dwave-qpu"
ALGORITHM = "quantum-anneal"
