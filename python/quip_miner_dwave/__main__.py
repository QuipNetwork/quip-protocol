"""``python -m quip_miner_dwave`` entry point."""
from __future__ import annotations

import sys

from quip_miner_dwave.cli import main

if __name__ == "__main__":
    sys.exit(main())
