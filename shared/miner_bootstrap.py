"""One-shot bootstrap flow for `quip-miner`.

Run by the `quip-miner bootstrap` subcommand. Idempotent — re-running against
an already-bootstrapped chain is a no-op that just verifies state.

Sequence:

1. Load or generate the signing keystore at `signer_key_path`.
2. Connect to the substrate node.
3. If `--seed-chain` is set (dev-only, see below): query `QuantumPow.Difficulty`
   and the snapshot API. If either is missing, submit
   `Sudo.sudo(QuantumPow.set_difficulty(...))` and
   `Sudo.sudo(QuantumPow.register_topology(...))` using a sudo signer
   (defaults to `//Alice` on a dev chain).
4. Query the miner's balance. If below `min_balance`, POST to the faucet.
   Wait until the chain reflects the new balance.
5. Query `QuantumPow.Miners[account]`. If not registered, submit
   `QuantumPow.register_miner()`.
6. Print a summary line.

The `--seed-chain` path is **dev-only**. `_assert_dev_chain` rejects
anything whose `system_chain` name doesn't match one of the
`Development` / `Local Testnet` / `quip-local` prefixes, and the
default sudo URI (`//Alice`) is the well-known dev key — any real
runtime configures `pallet_sudo::Key` to a quip-owned account, so a
mainnet bootstrap attempt would be rejected as a non-sudo origin even
if the chain-prefix guard were bypassed. Production deploys never
invoke `--seed-chain`: chainspec ships with topology + difficulty
baked in, and ad-hoc runtime config goes through ops tooling that
holds the real sudo key.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import dwave_networkx as dnx

from shared.allowed_value_spec import (
    AllowedValueSet,
    MILLI_SCALE as _MILLI_SCALE,
    scale_dict,
)
from shared.hybrid_signer import HybridSigner
from shared.keystore_hybrid import HybridKeystoreFile, load_or_generate
from shared.logging_config import get_logger
from substrate.client import SubstrateClient
from substrate.types import SubstrateDifficulty


# Default puzzle parameters seeded by `--seed-chain` on dev nodes. Mirror
# the runtime defaults documented in `quip-protocol-rs/runtime/src/configs`:
#   - h: ternary {-1, 0, +1} → 2 bits per node
#   - j: binary {-1, +1}     → 1 bit per edge
#   - spin: binary {-1, +1}  → 1 bit per spin (8x payload reduction)
_DEFAULT_ALLOWED_H = AllowedValueSet((-_MILLI_SCALE, 0, _MILLI_SCALE))
_DEFAULT_ALLOWED_J = AllowedValueSet((-_MILLI_SCALE, _MILLI_SCALE))
_DEFAULT_ALLOWED_SPIN = AllowedValueSet((-_MILLI_SCALE, _MILLI_SCALE))


# Dev chain names matched by prefix. The standalone faucet
# (`gitlab.com/quip.network/faucet`) keeps its own copy of this list; both
# stay loud-by-default about which chains they accept. `--chain=local3`
# reports "Local Testnet (3 Validators)" so a prefix match keeps the list short.
DEV_CHAIN_PREFIXES: Tuple[str, ...] = (
    "Development",
    "Local Testnet",
    "quip-local",
)


# Well-known dev master seeds for the hybrid signature scheme. Captured
# from `cargo run --example dump_dev_seeds -p quip-transaction-crypto`
# against `quip-protocol-rs` main. Verified byte-for-byte against the
# Rust-side `ALICE_PINNED_ACCOUNT_HEX` canary in genesis_config_presets.
# Update via the dump_dev_seeds example if the chain's dev mnemonic /
# derivation path ever change.
DEV_HYBRID_SEEDS = {
    "//Alice": bytes.fromhex(
        "e5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
    ),
    "//Bob": bytes.fromhex(
        "398f0c28f98885e046333d4a41c19cee4c37368a9832c6502f6cfd182e2aef89"
    ),
    "//Alice//stash": bytes.fromhex(
        "3c881bc4d45926680c64a7f9315eeda3dd287f8d598f3653d7c107799c5422b3"
    ),
}


def _resolve_dev_signer(uri: str) -> HybridSigner:
    """Return a HybridSigner for a known dev URI like //Alice.

    Substrate's URI-to-master-seed derivation (BIP39 + soft junctions) is
    non-trivial to replicate in Python; for dev URIs we use the precomputed
    seeds dumped from the Rust side. Anything outside `DEV_HYBRID_SEEDS`
    raises — production paymasters supply their own keystore.
    """
    if uri not in DEV_HYBRID_SEEDS:
        raise ValueError(
            f"unknown dev URI {uri!r}; known: {sorted(DEV_HYBRID_SEEDS)}. "
            "Load a hybrid keystore instead via shared.keystore_hybrid.load()."
        )
    return HybridSigner.from_master_seed(DEV_HYBRID_SEEDS[uri])


logger = get_logger("miner_bootstrap")


# Default seed topology: small Zephyr Z(2,2) for fast dev cycles. 80 nodes /
# 356 edges — comfortably above the chain's MinNodes=16 limit and small
# enough to mine in seconds on CPU.
DEFAULT_SEED_TOPOLOGY: Tuple[int, int] = (2, 2)

# Relaxed default difficulty, matching the legacy genesis defaults (CLAUDE.md
# "Genesis block defaults"). Easy to mine for dev validation; production
# operators should tune via `quip-miner set-difficulty` once that lands.
DEFAULT_SEED_DIFFICULTY = SubstrateDifficulty(
    min_solutions=5,
    max_energy_milli=-2_500_000,   # -2500.0
    min_diversity_milli=200,        # 0.2
)

# Minimum balance before bootstrap considers the account funded. Miner
# registration reserves `MinerDeposit = UNIT` (= 1e12 plancks on the dev
# chain), so we leave a comfortable cushion.
#
# "Planck" is the Substrate/Polkadot smallest-balance-unit term (analogous
# to `wei` on Ethereum or `satoshi` on Bitcoin). 1 UNIT = 10^12 plancks on
# 12-decimal chains.
DEFAULT_MIN_BALANCE_PLANCKS = 2_000_000_000_000  # 2 UNIT

# Amount the faucet sends per request.
DEFAULT_FAUCET_TOP_UP_PLANCKS = 10_000_000_000_000  # 10 UNIT

# Faucet funding is retried for this long before startup gives up. The
# faucet mints via `sudo(FaucetOps.mint)` and deliberately does not burn a
# caller's rate-limit slot on failure, so transient errors (a node blip, a
# stale websocket, the faucet still coming up alongside the chain) are meant
# to be retried. Budget is consumed by the planned backoff sleeps below.
FAUCET_FUNDING_TIMEOUT_SECONDS = 300.0  # 5 minutes
_FAUCET_BACKOFF_START_SECONDS = 2.0
_FAUCET_BACKOFF_MAX_SECONDS = 30.0

# HTTP statuses worth retrying: rate-limit + the 5xx family the faucet
# returns when a mint fails transiently. Any other 4xx means our request is
# malformed and will fail identically on every retry.
_RETRYABLE_FAUCET_STATUS = frozenset({429, 500, 502, 503, 504})


class FaucetTransientError(RuntimeError):
    """Faucet failure that may clear on retry (429 / 5xx / connection)."""


class FaucetPermanentError(RuntimeError):
    """Faucet failure that retrying cannot fix (malformed request → 4xx)."""


@dataclass
class BootstrapConfig:
    # Ordered failover list — `SubstrateClient` tries each in turn at
    # connect() and live-rotates through them if the connection drops
    # mid-bootstrap. A single-entry tuple is the legacy single-URL case.
    validators: Tuple[str, ...]
    signer_key_path: Path
    faucet_url: Optional[str] = None
    sudo_key_uri: str = "//Alice"
    seed_chain: bool = False
    seed_topology_mt: Tuple[int, int] = DEFAULT_SEED_TOPOLOGY
    seed_difficulty: SubstrateDifficulty = field(
        default_factory=lambda: DEFAULT_SEED_DIFFICULTY
    )
    # Force a sudo `set_difficulty` even when one is already on chain.
    # The runtime tightens difficulty between proofs (adjusting min_diversity
    # / max_energy upward when blocks arrive too fast); long-haul integration
    # tests need to reset to the relaxed seed each run or the CPU SA can't
    # keep up. Defaults False so the CLI's `bootstrap` subcommand stays
    # idempotent in production.
    force_reseed_difficulty: bool = False
    min_balance_plancks: int = DEFAULT_MIN_BALANCE_PLANCKS
    faucet_top_up_plancks: int = DEFAULT_FAUCET_TOP_UP_PLANCKS
    # Total wall budget (seconds) spent retrying a transient faucet failure
    # before startup gives up. Tunable mainly so tests can shrink it.
    faucet_timeout_seconds: float = FAUCET_FUNDING_TIMEOUT_SECONDS


@dataclass
class BootstrapResult:
    ss58_address: str
    account_id_hex: str
    balance_plancks: int
    miner_registered: bool
    topology_seeded: bool
    difficulty_seeded: bool


async def bootstrap(config: BootstrapConfig) -> BootstrapResult:
    keystore = load_or_generate(config.signer_key_path)
    logger.info(
        "using signing keystore: path=%s ss58=%s",
        keystore.path,
        keystore.signer.ss58_address(),
    )

    # Bootstrap is a one-shot: there's no long-running loop that benefits
    # from the pool's hot-active validator swap. A direct SubstrateClient
    # over the configured URL list (with its own walk-on-connect failover)
    # is enough, and avoids spawning a validator child process just to
    # tear it down moments later.
    client = SubstrateClient(urls=config.validators)
    await client.connect()
    try:
        topology_seeded = False
        difficulty_seeded = False
        if config.seed_chain:
            topology_seeded, difficulty_seeded = await _maybe_seed_chain(
                client, config
            )

        balance = await ensure_funded(client, keystore, config)
        # Idempotent: returns whether it *newly* registered; either way the
        # account is registered past this point (it raises on failure).
        await _ensure_registered(client, keystore)

        return BootstrapResult(
            ss58_address=keystore.signer.ss58_address(),
            account_id_hex="0x" + keystore.signer.account_id_bytes().hex(),
            balance_plancks=balance,
            miner_registered=True,
            topology_seeded=topology_seeded,
            difficulty_seeded=difficulty_seeded,
        )
    finally:
        await client.close()


# ----------------------------------------------------------------------
# Chain seeding (sudo path)
# ----------------------------------------------------------------------


async def _assert_dev_chain(client: SubstrateClient) -> None:
    """Refuse to issue sudo extrinsics against a non-dev chain.

    The module docstring advertises this guard; the chain-side
    `pallet_sudo::Key != //Alice` rejection is a backstop, not a
    substitute. A non-dev runtime that happens to have a known sudo key
    (misconfigured testnet, stale chainspec) would otherwise let
    `--seed-chain` mutate production state.
    """
    chain_name = await client._run(lambda: client._iface.chain)  # noqa: SLF001
    if not any(chain_name.startswith(p) for p in DEV_CHAIN_PREFIXES):
        raise RuntimeError(
            f"refusing --seed-chain against non-dev chain {chain_name!r}; "
            f"allowed prefixes: {', '.join(DEV_CHAIN_PREFIXES)}"
        )
    logger.info("bootstrap verified dev chain: %s", chain_name)


async def _maybe_seed_chain(
    client: SubstrateClient, config: BootstrapConfig
) -> Tuple[bool, bool]:
    """Submit sudo extrinsics to seed Difficulty + DefaultTopology if missing.

    Idempotent: returns `(topology_seeded, difficulty_seeded)` reflecting
    whether *this* call did the seeding. Both are False on re-runs.
    """
    await _assert_dev_chain(client)
    sudo_signer = _resolve_dev_signer(config.sudo_key_uri)

    difficulty_seeded = False
    needs_seed = (
        await client.query_difficulty() is None or config.force_reseed_difficulty
    )
    if needs_seed:
        if config.force_reseed_difficulty:
            # Defense in depth: `_assert_dev_chain` above already refused
            # non-dev chains, but log a loud warning when we're about to
            # overwrite an existing on-chain difficulty so the action is
            # never invisible. Production CLI doesn't expose the flag.
            logger.warning(
                "force-reseeding QuantumPow.Difficulty via sudo: "
                "min_solutions=%d max_energy_milli=%d min_diversity_milli=%d",
                config.seed_difficulty.min_solutions,
                config.seed_difficulty.max_energy_milli,
                config.seed_difficulty.min_diversity_milli,
            )
        else:
            logger.info("seeding QuantumPow.Difficulty via sudo (first time)")
        await _sudo_call(
            client,
            sudo_signer,
            "QuantumPow",
            "set_difficulty",
            {"difficulty": _difficulty_to_dict(config.seed_difficulty)},
        )
        difficulty_seeded = True
    else:
        logger.info("QuantumPow.Difficulty already set; skipping seed")

    topology_seeded = False
    # Probe whether DefaultTopology is set by asking for the snapshot. If the
    # runtime API returns None despite Difficulty being set, that's the
    # missing-topology case — see pallets/quantum-pow/src/lib.rs:457 where
    # mining_snapshot bails on DefaultTopology::get()? returning None.
    snapshot = await client.get_mining_snapshot(
        miner_account_bytes=b"\x00" * 32,  # placeholder; we just want the call
    )
    if snapshot is None:
        nodes, edges = _build_seed_topology(config.seed_topology_mt)
        logger.info(
            "seeding QuantumPow topology via sudo: %d nodes, %d edges",
            len(nodes),
            len(edges),
        )
        # `NodesOf<T>` / `EdgesOf<T>` are `BoundedVec<...>` which substrate
        # metadata exposes as a 1-field composite. scalecodec wants the inner
        # Vec wrapped in a single-element tuple — see the python_parity
        # encoder doc in py-substrate-interface for the composite shape rule.
        # Post-MR-!20 also passes the three `AllowedValueSpec` payloads
        # (h, j, spin) — the chain rejects topologies with an empty spec.
        await _sudo_call(
            client,
            sudo_signer,
            "QuantumPow",
            "register_topology",
            {
                "nodes": (nodes,),
                "edges": (edges,),
                "allowed_h_values": scale_dict(_DEFAULT_ALLOWED_H),
                "allowed_j_values": scale_dict(_DEFAULT_ALLOWED_J),
                "allowed_spin_values": scale_dict(_DEFAULT_ALLOWED_SPIN),
            },
        )
        topology_seeded = True
    else:
        logger.info(
            "QuantumPow topology already registered (hash=0x%s); skipping seed",
            snapshot.topology_hash.hex(),
        )

    return topology_seeded, difficulty_seeded


async def _sudo_call(
    client: SubstrateClient,
    sudo_signer: HybridSigner,
    inner_module: str,
    inner_function: str,
    inner_params: dict,
) -> None:
    """Compose and submit `Sudo.sudo(<inner_call>)`."""
    # substrate-interface lets us nest by composing the inner call first,
    # then passing it as the `call` param of Sudo.sudo.
    iface = client._iface  # noqa: SLF001 — bootstrap is internal client
    inner_call = await client._run(  # noqa: SLF001
        lambda: iface.compose_call(
            call_module=inner_module,
            call_function=inner_function,
            call_params=inner_params,
        )
    )
    receipt = await client.submit_extrinsic(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": inner_call.value},
        signer=sudo_signer,
        wait_for="inblock",
    )
    if receipt.error:
        raise RuntimeError(
            f"sudo call {inner_module}.{inner_function} failed: {receipt.error}"
        )


def _difficulty_to_dict(d: SubstrateDifficulty) -> dict:
    return {
        "min_solutions": d.min_solutions,
        "max_energy_milli": d.max_energy_milli,
        "min_diversity_milli": d.min_diversity_milli,
    }


def _build_seed_topology(mt: Tuple[int, int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Generate a Zephyr Z(m,t) graph using sampler-compatible node labels.

    The labels are whatever `dwave_networkx.zephyr_graph` assigns (linear ints,
    but typically non-contiguous depending on Zephyr's tile structure).
    Critically these are *the same* labels that
    `dwave_topologies.topologies.zephyr.ZephyrTopology` exposes via its
    `nodes`/`edges` properties, which is what miners' samplers use as
    coordinate keys when receiving `h` and `J` dicts.

    Earlier versions of this helper remapped to dense 0..n-1, on the
    mistaken belief the chain required it. The chain's
    `validate_topology_consistency` only forbids duplicate node ids — any
    `u32` label is fine. Remapping created a label mismatch between
    chain-registered topology and the miner's sampler, causing every proof
    to fail topology-hash verification.
    """
    m, t = mt
    g = dnx.zephyr_graph(m=m, t=t)
    nodes = sorted(int(n) for n in g.nodes())
    edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in g.edges()
    )
    return nodes, edges


# ----------------------------------------------------------------------
# Funding
# ----------------------------------------------------------------------


async def ensure_funded(
    client: SubstrateClient,
    keystore: HybridKeystoreFile,
    config: BootstrapConfig,
) -> int:
    account = keystore.signer.account_id_bytes()
    balance = await client.query_balance(account)
    if balance >= config.min_balance_plancks:
        logger.info(
            "miner account already funded: balance=%d plancks", balance
        )
        return balance

    if config.faucet_url is None:
        raise RuntimeError(
            f"account balance {balance} below threshold {config.min_balance_plancks} "
            "but no --faucet-url was provided"
        )

    logger.info(
        "requesting %d plancks from faucet for %s (retrying up to %.0fs)",
        config.faucet_top_up_plancks,
        keystore.signer.ss58_address(),
        config.faucet_timeout_seconds,
    )
    faucet_url = config.faucet_url  # narrowed: non-None past the guard above
    dest_hex = "0x" + account.hex()
    budget = config.faucet_timeout_seconds
    backoff = _FAUCET_BACKOFF_START_SECONDS
    last_note = "no faucet response yet"
    attempt = 0
    while True:
        attempt += 1
        # A FaucetPermanentError (malformed request) propagates straight out
        # — no amount of retrying fixes a bad dest/amount.
        balance, last_note = await _try_fund_once(
            client,
            account,
            faucet_url=faucet_url,
            dest_hex=dest_hex,
            amount=config.faucet_top_up_plancks,
        )
        if balance >= config.min_balance_plancks:
            logger.info(
                "faucet funded after %d attempt(s): balance=%d plancks",
                attempt, balance,
            )
            return balance
        if budget <= 0:
            break
        wait = min(backoff, budget)
        logger.warning(
            "faucet not funded yet (%s); attempt %d, retrying in %.1fs "
            "(%.0fs budget left)",
            last_note, attempt, wait, budget,
        )
        await asyncio.sleep(wait)
        budget -= wait
        backoff = min(backoff * 2.0, _FAUCET_BACKOFF_MAX_SECONDS)
    raise RuntimeError(
        f"faucet did not fund within {config.faucet_timeout_seconds:.0f}s "
        f"after {attempt} attempt(s); balance is still {balance} "
        f"(last status: {last_note})"
    )


async def _try_fund_once(
    client: SubstrateClient,
    account: bytes,
    *,
    faucet_url: str,
    dest_hex: str,
    amount: int,
) -> Tuple[int, str]:
    """One faucet POST + balance read for the ``ensure_funded`` retry loop.

    Returns ``(balance, status_note)``. A transient faucet failure is
    swallowed and surfaced in the note so the caller backs off and retries;
    the balance read still runs (it's the source of truth, and an earlier
    mint may have settled). ``FaucetPermanentError`` propagates so the caller
    fails fast.
    """
    note = "requested"
    try:
        _post_faucet(faucet_url, dest_hex=dest_hex, amount=amount)
    except FaucetTransientError as exc:
        # Includes 429: a mint to this dest is already in flight/done, so
        # don't treat re-requests as fatal — the balance read picks it up.
        note = str(exc)
    balance = await client.query_balance(account)
    return balance, note


def _post_faucet(url: str, *, dest_hex: str, amount: int) -> dict:
    """POST the faucet ``/request`` contract, classifying failures for retry.

    Raises :class:`FaucetTransientError` for failures that may clear on retry
    (429 / 5xx / connection / timeout) and :class:`FaucetPermanentError` for
    a malformed request (other 4xx). The contract is served by the standalone
    faucet at ``gitlab.com/quip.network/faucet``.
    """
    body = json.dumps({"dest": dest_hex, "amount": amount}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/request",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        msg = f"faucet returned {exc.code}: {detail}"
        if exc.code in _RETRYABLE_FAUCET_STATUS:
            raise FaucetTransientError(msg) from exc
        raise FaucetPermanentError(msg) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # HTTPError is caught above; this is connection-refused / DNS /
        # socket timeout — the faucet may just be slow to come up.
        raise FaucetTransientError(f"faucet unreachable: {exc}") from exc


# ----------------------------------------------------------------------
# Miner registration
# ----------------------------------------------------------------------


async def _ensure_registered(
    client: SubstrateClient,
    keystore: HybridKeystoreFile,
) -> bool:
    """Register the miner if it isn't already. Idempotent.

    Returns ``True`` when this call submitted ``register_miner`` (a fresh
    registration) and ``False`` when the account was already in
    ``QuantumPow.Miners`` — letting callers report "registered" vs "already
    registered" without a second chain query. Raises on a failed extrinsic.
    """
    account = keystore.signer.account_id_bytes()
    miner_info = await client.query_miner(account)
    if miner_info is not None:
        logger.info(
            "miner already registered: deposit=%d submitted=%d won=%d",
            miner_info.deposit,
            miner_info.proofs_submitted,
            miner_info.proofs_won,
        )
        return False

    logger.info("registering miner: ss58=%s", keystore.signer.ss58_address())
    receipt = await client.submit_extrinsic(
        call_module="QuantumPow",
        call_function="register_miner",
        call_params={},
        signer=keystore.signer,
        wait_for="inblock",
    )
    if receipt.error:
        raise RuntimeError(f"register_miner failed: {receipt.error}")
    return True


__all__ = [
    "BootstrapConfig",
    "BootstrapResult",
    "DEFAULT_SEED_DIFFICULTY",
    "DEFAULT_SEED_TOPOLOGY",
    "DEFAULT_MIN_BALANCE_PLANCKS",
    "DEFAULT_FAUCET_TOP_UP_PLANCKS",
    "FAUCET_FUNDING_TIMEOUT_SECONDS",
    "FaucetPermanentError",
    "FaucetTransientError",
    "bootstrap",
    "ensure_funded",
]
