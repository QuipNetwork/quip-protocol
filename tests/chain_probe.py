"""Shared chain-reachability probes for the live-chain test modules.

Four test modules each grew their own ``_chain_reachable`` TCP probe, which
answers "is something listening on 9944" and nothing more. That is the wrong
question for any test that seeds chain state: a developer running the Quip
Testnet stack locally has something listening, so the gate passes, and the
test then fails inside ``miner_bootstrap`` with

    RuntimeError: refusing --seed-chain against non-dev chain 'Quip Testnet'

which reads as a broken test suite rather than "these tests need a dev
chain." ``dev_chain_reachable`` asks the question the guard actually
enforces, so those tests skip instead of erroring.
"""

from __future__ import annotations

import socket

from substrate.miner_bootstrap import DEV_CHAIN_PREFIXES


def chain_reachable(url: str) -> bool:
    """True if a TCP connection to ``url``'s host:port succeeds.

    Enough for read-only tests. Use :func:`dev_chain_reachable` for anything
    that seeds or mutates chain state.
    """
    bare = url.split("://", 1)[1]
    host, _, port_str = bare.partition(":")
    port = int(port_str) if port_str else 9944
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


def dev_chain_reachable(url: str) -> bool:
    """True if ``url`` serves a chain that accepts dev-only bootstrap.

    Mirrors ``miner_bootstrap``'s own guard: the chain name must start with
    one of ``DEV_CHAIN_PREFIXES``. Any probe failure returns ``False`` so a
    flaky or unreachable node skips rather than erroring.
    """
    if not chain_reachable(url):
        return False
    try:
        from substrateinterface import SubstrateInterface

        chain_name = SubstrateInterface(url=url).chain
    except Exception:  # noqa: BLE001 — a probe failure means "skip"
        return False
    return any(chain_name.startswith(p) for p in DEV_CHAIN_PREFIXES)
