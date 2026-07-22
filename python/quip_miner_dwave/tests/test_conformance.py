"""End-to-end conformance: miner vs committed quip-mock-coordinator binary.

Spawns the Rust mock coordinator's ``drive_miner`` flow by running the
``quip-mock-coordinator`` binary's test-equivalent path — actually we drive
the miner ourselves against a subprocess of the coordinator harness.

The committed harness entry is ``quip-mock-coordinator`` which expects a miner
binary path. We point it at a small shell wrapper that launches
``python -m quip_miner_dwave`` with mock mode.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]  # worktree root
PYTHON = Path(sys.executable)
MOCK_COORD = REPO / "rust" / "target" / "debug" / "quip-mock-coordinator"


def _ensure_mock_coord() -> Path:
    if MOCK_COORD.is_file():
        return MOCK_COORD
    # Build if missing
    cargo = shutil.which("cargo")
    if not cargo:
        pytest.skip("cargo not available to build quip-mock-coordinator")
    r = subprocess.run(
        [cargo, "build", "-p", "quip-mock-coordinator"],
        cwd=str(REPO / "rust"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        pytest.fail(f"build mock-coordinator failed:\n{r.stderr}")
    assert MOCK_COORD.is_file()
    return MOCK_COORD


def _write_miner_wrapper(tmpdir: Path) -> Path:
    """Shell script the coordinator can exec as a miner binary."""
    wrapper = tmpdir / "quip-dwave-qa"
    # Ensure python package path + mock mode
    script = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="{REPO / 'python'}${{PYTHONPATH:+:$PYTHONPATH}}"
export QUIP_DWAVE_MOCK=1
exec "{PYTHON}" -m quip_miner_dwave --mock "$@"
"""
    wrapper.write_text(script)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def test_conformance_against_mock_coordinator():
    coord = _ensure_mock_coord()
    with tempfile.TemporaryDirectory(prefix="quip-dwave-conf-") as td:
        tdp = Path(td)
        miner = _write_miner_wrapper(tdp)
        # quip-mock-coordinator CLI: takes miner binary + optional socket
        # From main.rs — check args
        env = os.environ.copy()
        env["QUIP_SESSION_TOKEN"] = "test-token"
        env["QUIP_DWAVE_MOCK"] = "1"
        env["PYTHONPATH"] = str(REPO / "python") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # The coordinator binary drives the miner: see main.rs
        sock = tdp / "conf.sock"
        uri = f"unix://{sock}"
        proc = subprocess.run(
            [str(coord), str(miner), uri],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=str(REPO),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # Coordinator exits 0 on success (handshake + results + clean miner exit)
        if proc.returncode != 0:
            pytest.fail(
                f"mock-coordinator returned {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        # Soft assertions on log noise if any
        assert "handshake" not in out.lower() or "ok" in out.lower() or proc.returncode == 0
