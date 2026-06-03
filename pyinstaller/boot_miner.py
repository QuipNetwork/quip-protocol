# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""PyInstaller entry point for quip-miner."""

import multiprocessing
import sys

if __name__ == "__main__":
    # Handle --version immediately — no imports, no bundle extraction
    if "--version" in sys.argv:
        print("quip-protocol @VERSION@")
        sys.exit(0)

    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    from quip_cli import miner_main

    miner_main()
