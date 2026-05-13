"""Click-based CLI for quip-miner.

Provides the `quip-miner` console command with subcommands:
- keygen     - generate a fresh sr25519 signing key
- bootstrap  - fund + register a miner account against a substrate chain
- cpu / gpu / qpu - run a miner against a substrate chain

The legacy `quip-network-node` / `quip-network-simulator` commands were
removed in the v0.1 -> v0.2 refactor. Their state and orchestration are
now owned by the substrate node (quip-protocol-rs); miners attach via
SubstrateMinerController.
"""
from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Optional

import click

from shared.keystore import generate
from shared.logging_config import setup_logging
from shared.miner_bootstrap import BootstrapConfig, bootstrap


@click.group(name="quip-miner")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level for quip-miner subcommands",
)
def quip_miner(log_level: str) -> None:
    """Substrate-integrated quantum mining frontend."""
    setup_logging(log_level=log_level.upper(), node_name="quip-miner")


@quip_miner.command("keygen")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help="Where to write the generated keystore",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing keystore at --out")
def quip_miner_keygen(out_path: str, overwrite: bool) -> None:
    """Generate a fresh sr25519 signing key for quip-miner.

    Writes a JSON keystore (0o600) and prints the SS58 address. The seed is
    stored in plaintext — adequate for dev workflows where the faucet bot
    runs alongside. Passphrase-encrypted keystores land in Phase 7.
    """
    keystore = generate(Path(out_path).expanduser(), overwrite=overwrite)
    click.echo(f"wrote keystore: {keystore.path}")
    click.echo(f"ss58 address:   {keystore.signer.ss58_address()}")
    click.echo(f"account_id:     0x{keystore.signer.account_id_bytes().hex()}")


@quip_miner.command("bootstrap")
@click.option(
    "--node-url",
    required=True,
    help="Substrate node WebSocket URL (e.g. ws://localhost:9944)",
)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help="Path to the keystore (will be created if missing)",
)
@click.option(
    "--faucet-url",
    default=None,
    help="If set, request funding from this faucet bot when balance is low",
)
@click.option(
    "--seed-chain/--no-seed-chain",
    default=False,
    help="DEV-ONLY: on a fresh dev chain (Development / Local Testnet / "
    "quip-local), sudo-seed QuantumPow.Difficulty and "
    "QuantumPow.DefaultTopology. Refused against non-dev chains; never "
    "used in production (default off; requires --sudo-key)",
)
@click.option(
    "--sudo-key",
    "sudo_key_uri",
    default="//Alice",
    show_default=True,
    help="Substrate URI for the sudo signer (dev only)",
)
@click.option(
    "--seed-topology",
    "seed_topology_mt",
    default="2,2",
    show_default=True,
    help="Zephyr Z(m,t) parameters for the seed topology when --seed-chain is set",
)
def quip_miner_bootstrap(
    node_url: str,
    signer_key_path: str,
    faucet_url: Optional[str],
    seed_chain: bool,
    sudo_key_uri: str,
    seed_topology_mt: str,
) -> None:
    """Fund and register a miner account against a substrate chain.

    Idempotent — re-running against a fully-bootstrapped account is a no-op
    that just verifies state.
    """
    try:
        m_str, t_str = seed_topology_mt.split(",")
        topology_mt = (int(m_str), int(t_str))
    except (ValueError, AttributeError) as exc:
        raise click.BadParameter(
            f"--seed-topology must be 'm,t' (e.g. '2,2'), got {seed_topology_mt!r}"
        ) from exc

    config = BootstrapConfig(
        node_url=node_url,
        signer_key_path=Path(signer_key_path).expanduser(),
        faucet_url=faucet_url,
        sudo_key_uri=sudo_key_uri,
        seed_chain=seed_chain,
        seed_topology_mt=topology_mt,
    )
    result = asyncio.run(bootstrap(config))

    click.echo("bootstrap complete")
    click.echo(f"  ss58 address       : {result.ss58_address}")
    click.echo(f"  account_id         : {result.account_id_hex}")
    click.echo(f"  balance (plancks)  : {result.balance_plancks}")
    click.echo(f"  miner registered   : {result.miner_registered}")
    click.echo(f"  topology seeded    : {result.topology_seeded}")
    click.echo(f"  difficulty seeded  : {result.difficulty_seeded}")


# The faucet bot is shipped as a standalone script (`faucet_bot.py` at repo
# root) so it can be deployed independently of the rest of `quip-protocol`.
# Run with:
#
#     python faucet_bot.py --node-url ws://localhost:9944 --faucet-key //Alice
#
# See faucet_bot.py for the full CLI surface.


# --------------------------------------------------------------------------
# Mining subcommands: cpu / gpu / qpu spawn the controller end-to-end.
# Topology binding is explicit via --topology zephyr:M,T (defaults to Z(9,2)
# which matches the legacy chain difficulty calibration). The CLI verifies
# at startup that the sampler's topology hash matches the chain's snapshot;
# mismatch fails fast with the registered hash printed so the operator can
# fix --topology or re-seed the chain.
# --------------------------------------------------------------------------


def _parse_topology(spec: str):
    """Parse a `family:m,t` topology spec into a (`ZephyrTopology`, hash) pair.

    Only zephyr is supported in Phase 5a; pegasus/chimera land later if
    miners need them. Returns the sampler-compatible topology object so the
    CLI can plug it straight into the spec's `args["topology"]`.
    """
    from dwave_topologies.topologies.zephyr import zephyr

    if ":" not in spec:
        raise click.BadParameter(
            f"--topology must be 'family:m,t' (got {spec!r}); try 'zephyr:9,2'"
        )
    family, params = spec.split(":", 1)
    if family.lower() != "zephyr":
        raise click.BadParameter(
            f"only 'zephyr' topology is supported in Phase 5a (got {family!r})"
        )
    try:
        m_str, t_str = params.split(",")
        m, t = int(m_str), int(t_str)
    except ValueError as exc:
        raise click.BadParameter(
            f"--topology zephyr params must be 'm,t' (got {params!r})"
        ) from exc
    return zephyr(m, t)


async def _run_miner(
    *,
    kind: str,
    node_url: str,
    signer_key_path: str,
    rest_port: int,
    topology_spec: str,
    miner_config: dict,
):
    """Shared entry point for `quip-miner cpu|gpu|qpu`.

    Builds the keystore-loaded signer, a `MinerCore` with the requested
    miner kind, two `SubstrateClient` instances (state + subscription),
    and a `SubstrateMinerController`. Runs the controller until KeyboardInterrupt.
    """
    import asyncio
    import hashlib
    import signal as signal_module
    from pathlib import Path

    from shared.keystore import load
    from shared.miner_core import MinerCore
    from shared.signer import Sr25519Signer
    from shared.substrate_client import SubstrateClient
    from shared.substrate_miner_controller import SubstrateMinerController

    keystore = load(Path(signer_key_path).expanduser())
    click.echo(f"signer: {keystore.signer.ss58_address()}")

    # Build the sampler topology and bind it to the miner config.
    topology = _parse_topology(topology_spec)
    click.echo(
        f"topology: {topology_spec} ({topology.num_nodes} nodes, "
        f"{topology.num_edges} edges)"
    )

    # Inject the topology into whichever miner kind is being constructed.
    miner_config = _inject_topology(miner_config, kind, topology)

    core = MinerCore(node_id="quip-miner", miners_config=miner_config)
    if not core.miner_handles:
        click.echo(
            f"no miner handles built for kind={kind}; check --num-cpus / "
            f"GPU/QPU config",
            err=True,
        )
        return 2

    client = SubstrateClient(url=node_url)
    await client.connect()

    # Verify the sampler topology matches the chain's registered topology
    # before mining starts. Without this check, every proof would be rejected
    # with InvalidTopology (Phase 4 confirmed this end-to-end).
    head = await client.get_head()
    snapshot = await client.get_mining_snapshot(
        at=head,
        miner_account_bytes=keystore.signer.account_id_bytes(),
    )
    if snapshot is None:
        click.echo(
            "chain has no registered topology; run "
            "`quip-miner bootstrap --seed-chain` first",
            err=True,
        )
        await client.close()
        core.close()
        return 3
    expected_hash = _zephyr_topology_hash(topology)
    if snapshot.topology_hash != expected_hash:
        click.echo(
            f"topology mismatch: --topology {topology_spec} hashes to "
            f"0x{expected_hash.hex()} but chain has 0x{snapshot.topology_hash.hex()}; "
            "either adjust --topology or re-seed the chain",
            err=True,
        )
        await client.close()
        core.close()
        return 4

    controller = SubstrateMinerController(
        client=client,
        signer=keystore.signer,
        miner_handles=core.miner_handles,
        topology_hash=snapshot.topology_hash,
    )
    click.echo(
        f"controller starting: handles={[h.miner_id for h in core.miner_handles]} "
        f"topology_hash=0x{snapshot.topology_hash.hex()[:16]}..."
    )

    # Optional telemetry server. The legacy /api/v1/* surface is preserved
    # via `shared.telemetry_api`; new operators that don't want HTTP
    # telemetry pass `--rest-port -1` to skip the bind.
    telemetry: Optional["TelemetryApiServer"] = None
    if rest_port is not None and rest_port > 0:
        from shared.telemetry_api import TelemetryApiServer

        telemetry = TelemetryApiServer(
            core=core,
            client=client,
            signer=keystore.signer,
            controller=controller,
            host="127.0.0.1",
            port=rest_port,
        )
        await telemetry.start()
        click.echo(
            f"telemetry api: http://127.0.0.1:{rest_port}/api/v1/status"
        )

    loop = asyncio.get_running_loop()
    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        loop.add_signal_handler(sig, controller.shutdown)

    try:
        await controller.run()
    finally:
        click.echo(
            f"controller stopped: stats={controller.stats}",
        )
        if telemetry is not None:
            await telemetry.stop()
        await client.close()
        core.close()
    return 0


def _inject_topology(miner_config: dict, kind: str, topology) -> dict:
    """Stash the topology in the appropriate miner section's `args`.

    The miner-builder paths in `MinerCore._initialize_miners` forward
    `args["topology"]` straight to the sampler constructor.
    """
    out = dict(miner_config)
    if kind == "cpu":
        section = dict(out.get("cpu") or {})
        args = dict(section.get("args") or {})
        args["topology"] = topology
        section["args"] = args
        out["cpu"] = section
    # GPU/QPU sections are device-array shaped; topology injection happens
    # at the spec level via _build_gpu_specs / _build_qpu_specs in a
    # follow-on once GPU/QPU end-to-end is exercised against the chain.
    return out


def _zephyr_topology_hash(topology) -> bytes:
    """Compute the chain's blake2_256 topology hash from a Zephyr graph.

    Matches `pallets/quantum-pow/src/topology.rs::hash_topology`:
        blake2_256(SCALE((sorted nodes, sorted canonical edges)))
    """
    import hashlib

    nodes = sorted(int(n) for n in topology.nodes)
    edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in topology.edges
    )

    def compact_len(n: int) -> bytes:
        if n < 0x40:
            return bytes([n << 2])
        if n < 0x4000:
            return ((n << 2) | 0b01).to_bytes(2, "little")
        if n < 0x40000000:
            return ((n << 2) | 0b10).to_bytes(4, "little")
        raise ValueError(f"compact len {n} exceeds 30-bit range")

    buf = compact_len(len(nodes))
    for n in nodes:
        buf += n.to_bytes(4, "little")
    buf += compact_len(len(edges))
    for u, v in edges:
        buf += u.to_bytes(4, "little") + v.to_bytes(4, "little")
    return hashlib.blake2b(buf, digest_size=32).digest()


_NODE_URL_HELP = "Substrate node WebSocket URL (e.g. ws://localhost:9944)"
_SIGNER_KEY_HELP = "Path to the signing keystore (created by `quip-miner keygen`)"
_TOPOLOGY_HELP = (
    "Topology spec for the miner's sampler. Format: 'family:m,t'. Must hash "
    "to the chain's registered topology — mismatch fails fast at startup."
)
_REST_PORT_HELP = "Telemetry REST API port (set to -1 to disable; Phase 5b wires this in)"


@quip_miner.command("cpu")
@click.option("--node-url", required=True, help=_NODE_URL_HELP)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help=_SIGNER_KEY_HELP,
)
@click.option(
    "--num-cpus",
    type=int,
    default=1,
    show_default=True,
    help="Number of CPU SA workers to spawn",
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port", type=int, default=-1, show_default=True, help=_REST_PORT_HELP,
)
def quip_miner_cpu(
    node_url: str,
    signer_key_path: str,
    num_cpus: int,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run CPU SA miners against a substrate chain."""
    import asyncio

    miner_config = {"cpu": {"num_cpus": num_cpus}}
    raise SystemExit(asyncio.run(_run_miner(
        kind="cpu",
        node_url=node_url,
        signer_key_path=signer_key_path,
        rest_port=rest_port,
        topology_spec=topology_spec,
        miner_config=miner_config,
    )))


@quip_miner.command("gpu")
@click.option("--node-url", required=True, help=_NODE_URL_HELP)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help=_SIGNER_KEY_HELP,
)
@click.option(
    "--gpu-backend",
    type=click.Choice(["local", "metal", "modal"], case_sensitive=False),
    default="local",
    show_default=True,
    help="GPU backend: local CUDA, Apple Metal, or Modal cloud",
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port", type=int, default=-1, show_default=True, help=_REST_PORT_HELP,
)
def quip_miner_gpu(
    node_url: str,
    signer_key_path: str,
    gpu_backend: str,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run a GPU miner (CUDA / Metal / Modal) against a substrate chain.

    GPU end-to-end verification against the chain is a Phase 6 follow-on —
    the controller and spec wiring work, but `--topology` injection for
    GPU samplers landed in a single CPU-only path in Phase 5a. CUDA / Metal
    samplers consume the topology via their `args` dict the same way the
    CPU path does once that injection is generalised.
    """
    import asyncio

    backend = gpu_backend.lower()
    if backend == "local":
        miner_config = {"cuda": [{"device": "0"}]}
    elif backend == "metal":
        miner_config = {"metal": [{}]}
    elif backend == "modal":
        miner_config = {"modal": [{"gpu_type": "t4"}]}
    else:
        raise click.BadParameter(f"unknown --gpu-backend: {backend}")

    raise SystemExit(asyncio.run(_run_miner(
        kind="gpu",
        node_url=node_url,
        signer_key_path=signer_key_path,
        rest_port=rest_port,
        topology_spec=topology_spec,
        miner_config=miner_config,
    )))


@quip_miner.command("qpu")
@click.option("--node-url", required=True, help=_NODE_URL_HELP)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help=_SIGNER_KEY_HELP,
)
@click.option(
    "--qpu-type",
    type=click.Choice(["dwave", "ibm", "ionq", "pasqal", "braket", "origin"]),
    default="dwave",
    show_default=True,
    help="QPU provider",
)
@click.option(
    "--daily-budget",
    type=int,
    default=None,
    help="Daily QPU access-time budget in microseconds (provider-specific)",
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port", type=int, default=-1, show_default=True, help=_REST_PORT_HELP,
)
def quip_miner_qpu(
    node_url: str,
    signer_key_path: str,
    qpu_type: str,
    daily_budget,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run a QPU miner against a substrate chain.

    Provider credentials come from the environment (e.g. DWAVE_API_KEY).
    Same Phase 5a caveat as GPU: end-to-end against the chain is a Phase 6
    item once topology binding generalises beyond CPU.
    """
    import asyncio

    section: dict = {"type": qpu_type}
    if daily_budget is not None:
        section["daily_budget"] = daily_budget
    miner_config = {qpu_type: [section]}

    raise SystemExit(asyncio.run(_run_miner(
        kind="qpu",
        node_url=node_url,
        signer_key_path=signer_key_path,
        rest_port=rest_port,
        topology_spec=topology_spec,
        miner_config=miner_config,
    )))


# Entry points for console_scripts



def miner_main():
    """Entry point for the `quip-miner` console script."""
    quip_miner(standalone_mode=False)

