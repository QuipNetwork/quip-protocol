"""CLI for ``quip-dwave-qa`` / ``python -m quip_miner_dwave``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from quip_miner_dwave import (
    ALGORITHM,
    BACKEND,
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
    EXIT_INTERNAL_FATAL,
    __version__,
)
from quip_miner_dwave.ocean import (
    OceanSampler,
    credentials_present,
    mock_mode_enabled,
    ocean_importable,
)
from quip_miner_dwave.session_loop import run_session_sync


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quip-dwave-qa",
        description="QuIP D-Wave quantum-annealing miner (v0.3 protocol)",
    )
    p.add_argument("--quip-coordinator", default=None, help="unix:// path or host")
    p.add_argument("--miner-id", default="qpu-0")
    p.add_argument(
        "--capabilities",
        action="store_true",
        help="print capability JSON and exit",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="verify Ocean import + D-Wave credentials, then exit",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    p.add_argument(
        "--log-level",
        default="info",
        help="logging level (default: info)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="force offline mock sampler (also set by QUIP_DWAVE_MOCK=1)",
    )
    return p


def print_capabilities() -> None:
    print(
        json.dumps(
            {
                "backend": BACKEND,
                "algorithm": ALGORITHM,
                "supported_kinds": ["ISING_SAMPLE"],
                "max_nodes": 10000,
                "max_edges": 100000,
                "features": ["quantum-anneal", "native-topology"],
            }
        )
    )


def run_check(*, force_mock: bool = False) -> int:
    """``--check``: Ocean importable + (creds present OR mock mode)."""
    if not ocean_importable():
        print("FAIL: dwave-ocean-sdk / dimod not importable", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE
    if force_mock or mock_mode_enabled():
        print("OK: ocean importable; mock mode (no live QPU required)")
        return EXIT_CLEAN
    if not credentials_present():
        print(
            "FAIL: no D-Wave credentials "
            "(set DWAVE_API_KEY or configure ~/.config/dwave/dwave.conf)",
            file=sys.stderr,
        )
        return EXIT_ENV_INCOMPATIBLE
    print("OK: ocean importable; credentials present")
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.version:
        print(f"quip-dwave-qa {__version__} protocol 1")
        return EXIT_CLEAN
    if args.capabilities:
        print_capabilities()
        return EXIT_CLEAN
    if args.check:
        return run_check(force_mock=args.mock)

    if not args.quip_coordinator:
        print("--quip-coordinator required for session mode", file=sys.stderr)
        return EXIT_CONFIG_INVALID

    use_mock = args.mock or mock_mode_enabled()
    try:
        # *_file convention for the D-Wave token at startup (preferred over the
        # DWAVE_API_KEY .env literal, which stays a fallback). The coordinator's
        # config.toml api_token_file arrives via Configure after the connection
        # is made; wiring that requires deferring the connect (see quip-i6g.7).
        sampler = OceanSampler(
            mock=use_mock,
            api_token_file=os.environ.get("QUIP_DWAVE_API_TOKEN_FILE"),
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("sampler init failed: %s", exc)
        return EXIT_ENV_INCOMPATIBLE

    try:
        return run_session_sync(
            args.quip_coordinator,
            args.miner_id,
            sampler,
        )
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else EXIT_INTERNAL_FATAL
        return code
    except Exception:
        logging.getLogger(__name__).exception("fatal")
        return EXIT_INTERNAL_FATAL


if __name__ == "__main__":
    sys.exit(main())
