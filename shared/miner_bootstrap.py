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

from shared.hybrid_signer import HybridSigner
from shared.keystore_hybrid import HybridKeystoreFile, load_or_generate
from shared.logging_config import get_logger
from shared.substrate_client import SubstrateClient
from shared.substrate_types import SubstrateDifficulty


# Dev chain names matched by prefix. Mirrors the list in `faucet_bot.py`,
# which is intentionally standalone — duplicating here keeps both modules
# loud-by-default about which chains they accept. `--chain=local3` reports
# "Local Testnet (3 Validators)" so a prefix match keeps the list short.
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
    min_quality_milli=0,            # no quality floor on dev chain
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


@dataclass
class BootstrapConfig:
    node_url: str
    signer_key_path: Path
    faucet_url: Optional[str] = None
    sudo_key_uri: str = "//Alice"
    seed_chain: bool = False
    seed_topology_mt: Tuple[int, int] = DEFAULT_SEED_TOPOLOGY
    seed_difficulty: SubstrateDifficulty = field(
        default_factory=lambda: DEFAULT_SEED_DIFFICULTY
    )
    min_balance_plancks: int = DEFAULT_MIN_BALANCE_PLANCKS
    faucet_top_up_plancks: int = DEFAULT_FAUCET_TOP_UP_PLANCKS


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

    client = SubstrateClient(url=config.node_url)
    await client.connect()
    try:
        topology_seeded = False
        difficulty_seeded = False
        if config.seed_chain:
            topology_seeded, difficulty_seeded = await _maybe_seed_chain(
                client, config
            )

        balance = await _ensure_funded(client, keystore, config)
        miner_registered = await _ensure_registered(client, keystore)

        return BootstrapResult(
            ss58_address=keystore.signer.ss58_address(),
            account_id_hex="0x" + keystore.signer.account_id_bytes().hex(),
            balance_plancks=balance,
            miner_registered=miner_registered,
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
    if await client.query_difficulty() is None:
        logger.info("seeding QuantumPow.Difficulty via sudo")
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
        await _sudo_call(
            client,
            sudo_signer,
            "QuantumPow",
            "register_topology",
            {"nodes": (nodes,), "edges": (edges,)},
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
        "min_quality_milli": d.min_quality_milli,
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


async def _ensure_funded(
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
        "requesting %d plancks from faucet for %s",
        config.faucet_top_up_plancks,
        keystore.signer.ss58_address(),
    )
    _post_faucet(
        config.faucet_url,
        dest_hex="0x" + account.hex(),
        amount=config.faucet_top_up_plancks,
    )

    # Wait for the transfer to settle. With 6s slots and inclusion guaranteed
    # by `wait_for=inblock` on the faucet side, the balance should reflect
    # within one round trip plus one block.
    for attempt in range(10):
        await asyncio.sleep(2.0)
        balance = await client.query_balance(account)
        if balance >= config.min_balance_plancks:
            logger.info(
                "faucet settled after %.1fs: balance=%d plancks",
                (attempt + 1) * 2.0,
                balance,
            )
            return balance
    raise RuntimeError(
        f"faucet transfer did not settle within ~20s; balance is still {balance}"
    )


def _post_faucet(url: str, *, dest_hex: str, amount: int) -> dict:
    """Synchronous POST to the faucet. Kept blocking — we wait on chain after."""
    body = json.dumps({"dest": dest_hex, "amount": amount}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/faucet",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"faucet returned {exc.code}: {detail}"
        ) from exc


# ----------------------------------------------------------------------
# Miner registration
# ----------------------------------------------------------------------


async def _ensure_registered(
    client: SubstrateClient,
    keystore: HybridKeystoreFile,
) -> bool:
    account = keystore.signer.account_id_bytes()
    miner_info = await client.query_miner(account)
    if miner_info is not None:
        logger.info(
            "miner already registered: deposit=%d submitted=%d won=%d",
            miner_info.deposit,
            miner_info.proofs_submitted,
            miner_info.proofs_won,
        )
        return True

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
    "bootstrap",
]
