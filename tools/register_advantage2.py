"""Register the full Advantage2_system1 topology on a running QUIP chain via sudo.

The built-in ``quip-miner bootstrap --seed-chain`` path can't do this: it
refuses any chain whose name isn't a dev prefix (this chain reports
"Quip Testnet"), and it only seeds a *synthetic* Zephyr ``Z(m,t)`` graph.
This script submits the sudo calls in the required order:

  1. Compute ``target_hash`` (before any topology operations).
  2. ``Sudo.sudo(QuantumPow.register_topology(...))`` — the real 4,578-node /
     41,531-edge Advantage2_system1 hardware graph (fits the runtime's
     5,000-node / 50,000-edge BoundedVec bounds).
  3. ``Sudo.sudo(QuantumPow.add_mineable_topology(...))`` — whitelist the
     topology (required; ``set_default_topology`` rejects non-whitelisted
     topologies with ``TopologyNotMineable``).
  4. ``Sudo.sudo(QuantumPow.set_difficulty(...))`` — easy test config; must
     pass ``topology_hash`` now that difficulty is per-topology.
  5. ``Sudo.sudo(QuantumPow.set_default_topology(...))`` — repoint the active
     mining problem (topology must be registered AND whitelisted).

It registers the *exact* node/edge labels that ``load_topology(
"Advantage2_system1")`` returns — the same object a miner loads — so the
chain's ``hash_topology`` and the miner's ``compute_topology_hash`` agree by
construction. Finally it re-reads the snapshot through the miner's own
``resolve_topology_binding`` and asserts the local↔chain hashes match.

DEV/OPS ONLY. This deliberately skips ``_assert_dev_chain`` (which would
reject "Quip Testnet"); the //Alice dev sudo key only works because this is a
chain you own. Do not point it at anything you don't control.

Run (this is a state-mutating sudo write — review first):

    PYTHONPATH=. .quip/bin/python tools/register_advantage2.py \
        --validator ws://localhost:9944

Idempotent: a re-run skips topology registration if one is already present
and just re-asserts the binding.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bip39 import bip39_to_mini_secret
from dwave_topologies.topologies.json_loader import load_topology
from shared.hybrid_signer import HybridSigner
from shared.logging_config import get_logger
from substrate.client import SubstrateClient
from substrate.miner_bootstrap import (
    _DEFAULT_ALLOWED_H,
    _DEFAULT_ALLOWED_J,
    _DEFAULT_ALLOWED_SPIN,
    _difficulty_to_dict,
    _resolve_dev_signer,
    _sudo_call,
)
from shared.allowed_value_spec import AllowedValueSet, scale_dict
from shared.topology_hash import topology_hash as compute_topology_hash
from substrate.types import SubstrateDifficulty

logger = get_logger(__name__)

# Selectable linear-field (h) specs. "ternary" is the legacy v0.2 default
# (h ∈ {-1,0,+1}); "zero" is the J-only zero-field class (h ≡ 0) the testnet
# is moving to — its GSE curve is the protocol's baseline, so the difficulty
# energy target tracks what the QPU actually lands. Requires rc11+ on miners
# (zero-field gauge-fix + h-aware difficulty band) before repointing.
ALLOWED_H_SPECS = {
    "ternary": _DEFAULT_ALLOWED_H,  # AllowedValueSet((-1000, 0, 1000))
    "zero": AllowedValueSet((0,)),  # h ≡ 0
}

# Difficulty seeded when none exists or --force-difficulty is set. min_solutions
# and the diversity floor stay permissive; only the energy target varies (via
# --max-energy-milli) because that's what must track the active topology's GSE
# landscape. Default -2500.0 (easy). For h=0 use a value just easier than the
# zero-field reachable floor (e.g. -14_000_000 = -14000.0).
_DEFAULT_MAX_ENERGY_MILLI = -2_500_000


def _seed_difficulty(max_energy_milli: int) -> SubstrateDifficulty:
    return SubstrateDifficulty(
        min_solutions=1,
        max_energy_milli=max_energy_milli,
        min_diversity_milli=0,  # diversity gate disabled
    )


# Placeholder account for the read-only snapshot probe; the call just needs
# 32 bytes, the value doesn't affect what's registered.
PROBE_ACCOUNT = bytes(32)


def _resolve_signer(sudo_uri: str, mnemonic_file: str | None) -> HybridSigner:
    """Build the sudo HybridSigner from a mnemonic file or a dev URI.

    A real testnet (e.g. "Quip Testnet") configures ``pallet_sudo::Key`` to a
    quip-owned account, so the dev ``//Alice`` table doesn't apply. When
    ``mnemonic_file`` is given, derive the signer the same way substrate does:
    ``bip39_to_mini_secret(mnemonic)`` yields the 32-byte master seed that
    :meth:`HybridSigner.from_master_seed` HKDF-expands into the classical + PQ
    sub-keys. Verified to reproduce operator-1's ``5GZMoWFM…`` account.
    """
    if mnemonic_file:
        mnem = Path(mnemonic_file).expanduser().read_text().strip()
        master_seed = bytes(bip39_to_mini_secret(mnem, ""))
        return HybridSigner.from_master_seed(master_seed)
    return _resolve_dev_signer(sudo_uri)


async def _assert_is_sudo(client: SubstrateClient, signer: HybridSigner) -> None:
    """Refuse to submit unless the signer is the chain's ``Sudo.Key``.

    Without this, a wrong key sails through ``_sudo_call`` (the inner call
    fails with ``RequireSudo`` but ``Sudo.sudo`` succeeds at the extrinsic
    level, so nothing raises) and the operator thinks the op worked when it
    silently did nothing.
    """
    key = await client._run(lambda: client._iface.query("Sudo", "Key"))  # noqa: SLF001
    onchain = key.value if key else None
    mine = signer.ss58_address()
    if onchain != mine:
        raise SystemExit(
            f"signer {mine} is NOT the chain's Sudo.Key ({onchain}); "
            "sudo calls would fail with RequireSudo. Point --sudo-mnemonic at "
            "the operator key that owns Sudo.Key."
        )
    logger.info("sudo signer verified: %s == on-chain Sudo.Key", mine)


async def _topology_registered(client: SubstrateClient, topo_hash: bytes) -> bool:
    """True if ``topo_hash`` exists in the ``RegisteredTopologies`` map.

    Topologies are keyed by hash and coexist; ``register_topology`` errors
    ``TopologyAlreadyRegistered`` on a repeat. We check first so a re-run is
    idempotent rather than relying on the (swallowed) sudo inner error.
    """
    res = await client._run(  # noqa: SLF001
        lambda: client._iface.query(
            "QuantumPow", "RegisteredTopologies", ["0x" + topo_hash.hex()]
        )
    )
    return res is not None and res.value is not None


async def _default_topology_hash(client: SubstrateClient) -> bytes | None:
    """Current ``DefaultTopology`` pointer (the active mining topology), or None."""
    res = await client._run(  # noqa: SLF001
        lambda: client._iface.query("QuantumPow", "DefaultTopology")
    )
    if res is None or res.value is None:
        return None
    return bytes.fromhex(str(res.value).removeprefix("0x"))


def _normalize(topo) -> tuple[list[int], list[tuple[int, int]]]:
    """Sorted node ids + canonical (min,max) sorted edges.

    Mirrors ``miner_bootstrap._build_seed_topology`` exactly. ``topology_hash``
    canonicalizes the same way, so ordering doesn't change the hash — this just
    keeps the registered payload tidy and matches the proven seed path. Node
    labels are kept as-is (NOT remapped to 0..n-1): the miner's sampler keys h/J
    dicts by these labels, so remapping would break proof verification.
    """
    nodes = sorted(int(n) for n in topo.nodes)
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in topo.edges)
    return nodes, edges


async def main(
    url: str,
    sudo_uri: str,
    mnemonic_file: str | None,
    force_difficulty: bool,
    h_spec: str,
    max_energy_milli: int,
) -> None:
    client = SubstrateClient(url=url)
    await client.connect()
    try:
        chain = await client._run(lambda: client._iface.chain)  # noqa: SLF001
        logger.info("connected to %s (chain=%r)", url, chain)
        signer = _resolve_signer(sudo_uri, mnemonic_file)
        await _assert_is_sudo(client, signer)

        # --- 1. Compute target hash (must precede all topology operations) --
        # The hash is needed to check registration, whitelist, difficulty, and
        # set_default_topology — compute it once up front.
        allowed_h = ALLOWED_H_SPECS[h_spec]
        topo = load_topology("Advantage2_system1")
        nodes, edges = _normalize(topo)
        # The hash the chain will compute and a miner will independently derive
        # — same recipe as the runtime's hash_topology / the miner's binding.
        target_hash = compute_topology_hash(
            nodes, edges, allowed_h, _DEFAULT_ALLOWED_J, _DEFAULT_ALLOWED_SPIN
        )
        logger.info(
            "target topology: Advantage2_system1 h_spec=%s %d nodes %d edges hash=0x%s",
            h_spec,
            len(nodes),
            len(edges),
            target_hash.hex(),
        )

        # --- 2. Register topology (required before set_difficulty) ----------
        # set_difficulty now ensure!s the topology is in RegisteredTopologies.
        if await _topology_registered(client, target_hash):
            logger.info("topology already registered; skipping register_topology")
        else:
            logger.info(
                "registering topology via sudo (~%d KB extrinsic)...",
                (len(nodes) * 4 + len(edges) * 8) // 1024,
            )
            await _sudo_call(
                client,
                signer,
                "QuantumPow",
                "register_topology",
                {
                    "nodes": (nodes,),
                    "edges": (edges,),
                    "allowed_h_values": scale_dict(allowed_h),
                    "allowed_j_values": scale_dict(_DEFAULT_ALLOWED_J),
                    "allowed_spin_values": scale_dict(_DEFAULT_ALLOWED_SPIN),
                },
            )
            if not await _topology_registered(client, target_hash):
                raise SystemExit(
                    "register_topology did not take effect — the sudo extrinsic "
                    "was included but its inner call failed (check the Sudid "
                    "event, e.g. InvalidTopology / spec validation)."
                )
            logger.info("register_topology landed")

        # --- 3. Whitelist the target (required before set_default_topology) -
        # set_default_topology now rejects topologies not in MineableTopologies.
        # The first registered topology is auto-whitelisted by the runtime, but
        # subsequent topologies must be explicitly added. We always check and
        # add if absent — it is idempotent from the operator's perspective.
        mineable = await client.query_mineable_topologies()
        if target_hash in mineable:
            logger.info("topology already in MineableTopologies; skipping whitelist")
        else:
            logger.info(
                "whitelisting topology 0x%s via add_mineable_topology",
                target_hash.hex(),
            )
            await _sudo_call(
                client,
                signer,
                "QuantumPow",
                "add_mineable_topology",
                {"topology_hash": "0x" + target_hash.hex()},
            )
            mineable_after = await client.query_mineable_topologies()
            if target_hash not in mineable_after:
                raise SystemExit(
                    "add_mineable_topology did not take effect — inner call "
                    "failed (TopologyNotRegistered?). Whitelist unchanged."
                )
            logger.info("topology whitelisted")

        # --- 4. Difficulty (topology must be registered; hash is now known) --
        seed = _seed_difficulty(max_energy_milli)
        existing = await client.query_difficulty(target_hash)
        if existing is None or force_difficulty:
            logger.info(
                "setting difficulty via sudo: min_solutions=%d "
                "max_energy_milli=%d min_diversity_milli=%d",
                seed.min_solutions,
                seed.max_energy_milli,
                seed.min_diversity_milli,
            )
            await _sudo_call(
                client,
                signer,
                "QuantumPow",
                "set_difficulty",
                {
                    "topology_hash": "0x" + target_hash.hex(),
                    "difficulty": _difficulty_to_dict(seed),
                },
            )
            after = await client.query_difficulty(target_hash)
            if after is None or after.max_energy_milli != max_energy_milli:
                raise SystemExit(
                    "set_difficulty did not take effect — the sudo extrinsic "
                    "was included but its inner call failed (check the Sudid "
                    f"event). Wanted max_energy_milli={max_energy_milli}, "
                    f"chain has {after.max_energy_milli if after else None}."
                )
            logger.info("difficulty set (max_energy_milli=%d)", max_energy_milli)
        else:
            logger.info(
                "difficulty already set (max_energy_milli=%d); pass "
                "--force-difficulty to overwrite",
                existing.max_energy_milli,
            )

        # --- 5. Repoint DefaultTopology (topology registered + whitelisted) --
        # This is the consensus-affecting step: it changes the active mining
        # problem for the whole network and re-bases the difficulty energy curve.
        current_default = await _default_topology_hash(client)
        if current_default == target_hash:
            logger.info("DefaultTopology already points at target; no repoint")
        else:
            logger.info(
                "repointing DefaultTopology 0x%s -> 0x%s via sudo",
                current_default.hex() if current_default else "none",
                target_hash.hex(),
            )
            await _sudo_call(
                client,
                signer,
                "QuantumPow",
                "set_default_topology",
                {"topology_hash": "0x" + target_hash.hex()},
            )
            if await _default_topology_hash(client) != target_hash:
                raise SystemExit(
                    "set_default_topology did not take effect — inner call "
                    "failed (TopologyNotRegistered or TopologyNotMineable?). "
                    "DefaultTopology unchanged."
                )
            logger.info("DefaultTopology repointed")

        # --- 6. Verify via the miner's own binding path --------------------
        binding = await client.resolve_topology_binding(
            topo, miner_account_bytes=PROBE_ACCOUNT
        )
        print("\n=== result ===")
        print(f"h_spec              : {h_spec}")
        print(f"chain topology hash : 0x{binding.chain_hash.hex()}")
        print(f"local sampler hash  : 0x{binding.expected_hash.hex()}")
        print(f"hashes match        : {binding.matches}")
        if not binding.matches or binding.chain_hash != target_hash:
            raise SystemExit(
                "MISMATCH — chain default != target topology. A miner started "
                "with --topology Advantage2_system1 would fail fast."
            )
        print("\nOK. Restart miners (--topology Advantage2_system1) to rebind.")
        print(
            "NOTE: whitelisting (add_mineable_topology) is mandatory before "
            "set_default_topology on this runtime version. "
            "Re-baseline set_difficulty for the new energy curve if "
            "the GSE landscape changed materially (e.g. ternary -> zero)."
        )
    finally:
        await client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validator", default="ws://localhost:9944", help="WS RPC URL")
    ap.add_argument(
        "--sudo-mnemonic",
        default=None,
        help="path to a BIP39 mnemonic file for the real sudo key (e.g. "
        "../quip-protocol-rs/quip-testnet-keys/operator-1/mnemonic). When set, "
        "overrides --sudo-key.",
    )
    ap.add_argument(
        "--sudo-key",
        default="//Alice",
        help="dev sudo URI (used only when --sudo-mnemonic is absent)",
    )
    ap.add_argument(
        "--force-difficulty",
        action="store_true",
        help="overwrite an already-set difficulty with --max-energy-milli "
        "(min_solutions=1, diversity=0)",
    )
    ap.add_argument(
        "--max-energy-milli",
        type=int,
        default=_DEFAULT_MAX_ENERGY_MILLI,
        help="energy target (milli) to seed when difficulty is unset or "
        f"--force-difficulty (default {_DEFAULT_MAX_ENERGY_MILLI} = -2500.0). "
        "For h=0 use ~ -14_000_000 (just easier than the zero-field floor).",
    )
    ap.add_argument(
        "--h-spec",
        choices=sorted(ALLOWED_H_SPECS),
        default="ternary",
        help="linear-field spec to register/repoint to: 'ternary' (legacy "
        "h in {-1,0,1}) or 'zero' (J-only h=0). 'zero' requires rc11+ miners.",
    )
    args = ap.parse_args()
    asyncio.run(
        main(
            args.validator,
            args.sudo_key,
            args.sudo_mnemonic,
            args.force_difficulty,
            args.h_spec,
            args.max_energy_milli,
        )
    )
