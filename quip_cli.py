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
import hashlib
import signal
import traceback
from pathlib import Path
from typing import Optional

import click

from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate, load
from shared.logging_config import setup_logging
from shared.mempool_miner_controller import (
    MempoolMinerController,
    topology_hash_from_nodes_edges,
)
from shared.mempool_types import MinerType
from shared.miner_bootstrap import BootstrapConfig, bootstrap
from shared.miner_core import MinerCore
from shared.substrate_client import SubstrateClient
from shared.substrate_miner_controller import SubstrateMinerController
from shared.telemetry_api import TelemetryApiServer


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
    """Generate a fresh hybrid sr25519+ML-DSA-44 signing key for quip-miner.

    Writes a JSON keystore (0o600) holding the 32-byte master seed and
    prints the SS58 address. The seed is stored in plaintext — adequate
    for dev workflows where the faucet bot runs alongside.
    Passphrase-encrypted keystores land in a follow-on.
    """
    keystore = generate(Path(out_path).expanduser(), overwrite=overwrite)
    click.echo(f"wrote hybrid keystore: {keystore.path}")
    click.echo(f"ss58 address:          {keystore.signer.ss58_address()}")
    click.echo(f"account_id:            0x{keystore.signer.account_id_bytes().hex()}")
    click.echo(
        f"public key (1344B):    0x{keystore.signer.public_bytes().hex()[:48]}..."
    )


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


@quip_miner.command("register-solver")
@click.option(
    "--node-url",
    required=True,
    help="Substrate node WebSocket URL (e.g. ws://localhost:9944)",
)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False, exists=True),
    required=True,
    help="Path to the signing keystore (run `quip-miner keygen` first)",
)
@click.option(
    "--miner-type",
    type=click.Choice(
        ["cpu", "gpu", "qpu", "qpu_ibm", "qpu_ionq", "qpu_pasqal", "asic"],
        case_sensitive=False,
    ),
    required=True,
    help="Solver hardware family to register under "
    "(QuantumComputeMempool uses this for mode=Bid miner_type filtering). "
    "Bare 'qpu' maps to QpuDwave to match the legacy CPU/GPU/QPU triplet.",
)
def quip_miner_register_solver(
    node_url: str,
    signer_key_path: str,
    miner_type: str,
) -> None:
    """Register the keystore's account as a QuantumComputeMempool solver.

    One-time setup (idempotent if already registered). The solver type is
    surfaced to job proposers via `mode = Bid{miner_types: [...]}` filters.
    Use `quip-miner deregister-solver` to opt out.
    """
    keystore = load(Path(signer_key_path).expanduser())
    mt = MinerType.from_kind(miner_type)

    async def _do() -> int:
        client = SubstrateClient(url=node_url)
        try:
            await client.connect()
            existing = await client.query_solver(keystore.signer.account_id_bytes())
            if existing is not None:
                if existing.solver_type != mt:
                    click.echo(
                        f"solver already registered as {existing.solver_type.name}; "
                        f"deregister first to change to {mt.name}",
                        err=True,
                    )
                    return 4
                click.echo(
                    f"already registered as {mt.name} "
                    f"(submitted={existing.solutions_submitted}, "
                    f"earned={existing.rewards_earned} plancks)"
                )
                return 0

            receipt = await client.submit_extrinsic(
                "QuantumComputeMempool",
                "register_solver",
                {"solver_type": mt.to_scale_variant()},
                keystore.signer,
                wait_for="inblock",
            )
            if receipt.error:
                click.echo(f"register_solver failed: {receipt.error}", err=True)
                return 3
            click.echo(
                f"registered as {mt.name} "
                f"(extrinsic={receipt.extrinsic_hash}, block={receipt.block_hash})"
            )
            return 0
        finally:
            await client.close()

    raise SystemExit(asyncio.run(_do()))


@quip_miner.command("deregister-solver")
@click.option("--node-url", required=True, help="Substrate node WebSocket URL")
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False, exists=True),
    required=True,
    help="Path to the signing keystore",
)
def quip_miner_deregister_solver(node_url: str, signer_key_path: str) -> None:
    """Deregister the keystore's solver from QuantumComputeMempool.

    Idempotent against an unregistered account (returns 0 with a no-op
    message). After deregistration, submit_solution / claim_reward
    extrinsics will fail with `SolverNotRegistered` until you re-register.
    """
    keystore = load(Path(signer_key_path).expanduser())

    async def _do() -> int:
        client = SubstrateClient(url=node_url)
        try:
            await client.connect()
            existing = await client.query_solver(keystore.signer.account_id_bytes())
            if existing is None:
                click.echo("solver not registered; nothing to do")
                return 0

            receipt = await client.submit_extrinsic(
                "QuantumComputeMempool",
                "deregister_solver",
                {},
                keystore.signer,
                wait_for="inblock",
            )
            if receipt.error:
                click.echo(
                    f"deregister_solver failed: {receipt.error}", err=True
                )
                return 3
            click.echo(
                f"deregistered (extrinsic={receipt.extrinsic_hash}, "
                f"block={receipt.block_hash})"
            )
            return 0
        finally:
            await client.close()

    raise SystemExit(asyncio.run(_do()))


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
    else:
        click.echo(
            f"warning: --topology injection not yet implemented for kind={kind!r}; "
            "GPU/QPU samplers will use their default topology (Phase 6)",
            err=True,
        )
    return out


def _zephyr_topology_hash(topology) -> bytes:
    """Compute the chain's blake2_256 topology hash from a Zephyr graph.

    Matches `pallets/quantum-pow/src/topology.rs::hash_topology`:
        blake2_256(SCALE((sorted nodes, sorted canonical edges)))
    """
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
_REST_PORT_HELP = (
    "Telemetry REST API port (default -1 disables; set to a port to serve /api/v1/*)"
)

# Seconds to wait for a controller to drain after signalling shutdown.
# Must exceed the longest controller poll interval (<=10s for head-subscription
# timeout in SubstrateMinerController).
_SHUTDOWN_GRACE_SECONDS = 15.0


async def _run_concurrent_miner(
    *,
    mode: str,
    miner_kind: str,
    node_url: str,
    signer_key_path: str,
    rest_port: int,
    topology_spec: str,
    miner_config: dict,
) -> int:
    """Unified entry for `--mode pow|mempool|both` on cpu/gpu/qpu.

    Worker handle allocation rule:
      - mode=pow:     all handles → SubstrateMinerController
      - mode=mempool: all handles → MempoolMinerController
      - mode=both:    half (rounded down) to PoW, remainder to mempool.
                      Requires ≥2 handles; fails fast otherwise.

    Phase 8d's split is static and per-handle — no dynamic rebalancing.
    Phase 9 can introduce a shared scheduler that lets one mode use the
    other's idle worker, plus `--mempool-fraction` for non-half splits.
    """
    mode = mode.lower()
    if mode not in ("pow", "mempool", "both"):
        click.echo(f"invalid --mode {mode!r}", err=True)
        return 2

    keystore = load(Path(signer_key_path).expanduser())
    click.echo(f"signer: {keystore.signer.ss58_address()} (hybrid) mode={mode}")

    topology = _parse_topology(topology_spec)
    click.echo(
        f"topology: {topology_spec} ({topology.num_nodes} nodes, "
        f"{topology.num_edges} edges)"
    )

    miner_config = _inject_topology(miner_config, miner_kind, topology)
    core = MinerCore(node_id=f"quip-miner-{mode}", miners_config=miner_config)
    if not core.miner_handles:
        click.echo(f"no miner handles built for kind={miner_kind}", err=True)
        return 2

    pow_handles, mempool_handles = _split_handles_for_mode(
        mode, core.miner_handles
    )
    if mode == "both" and (not pow_handles or not mempool_handles):
        click.echo(
            f"--mode both requires at least 2 worker handles to split; "
            f"got {len(core.miner_handles)}. Increase --num-cpus / GPU "
            f"devices.",
            err=True,
        )
        core.close()
        return 5

    # Declare all network resources before the try so the finally block can
    # always clean up whatever was partially constructed on setup failure.
    client = None
    mempool_client = None
    telemetry = None
    pow_controller = None
    mempool_controller = None

    try:
        client = SubstrateClient(url=node_url)
        await client.connect()

        if pow_handles:
            # PoW requires the sampler's topology to match the chain's
            # registered DefaultTopology (the chain validates this in
            # `submit_proof` via `InvalidTopology`).
            try:
                head = await client.get_head()
                snapshot = await client.get_mining_snapshot(
                    at=head,
                    miner_account_bytes=keystore.signer.account_id_bytes(),
                )
            except Exception as exc:
                click.echo(
                    f"PoW mode: failed to query chain state: {exc}", err=True
                )
                return 3
            if snapshot is None:
                click.echo(
                    "PoW mode: chain has no registered topology; run "
                    "`quip-miner bootstrap --seed-chain` first",
                    err=True,
                )
                return 3
            expected_hash = _zephyr_topology_hash(topology)
            if snapshot.topology_hash != expected_hash:
                click.echo(
                    f"PoW mode topology mismatch: --topology {topology_spec} "
                    f"hashes to 0x{expected_hash.hex()} but chain has "
                    f"0x{snapshot.topology_hash.hex()}",
                    err=True,
                )
                return 4
            pow_topology_hash = snapshot.topology_hash
            pow_controller = SubstrateMinerController(
                client=client,
                signer=keystore.signer,
                miner_handles=pow_handles,
                topology_hash=pow_topology_hash,
                core=core,
            )
            click.echo(
                f"  pow handles: {[h.miner_id for h in pow_handles]} "
                f"topology=0x{pow_topology_hash.hex()[:16]}..."
            )

        if mempool_handles:
            sampler_nodes = tuple(int(n) for n in topology.nodes)
            sampler_edges = tuple((int(u), int(v)) for u, v in topology.edges)
            mempool_topology_hash = topology_hash_from_nodes_edges(
                sampler_nodes, sampler_edges
            )
            # In `both` mode the mempool controller needs its own client to
            # avoid serializing submissions behind the PoW client's asyncio
            # lock. In mempool-only mode `client` has no contention -- reuse it.
            mempool_client_ref = client
            if pow_handles:
                mempool_client = SubstrateClient(url=node_url)
                await mempool_client.connect()
                mempool_client_ref = mempool_client
            mempool_controller = MempoolMinerController(
                client=mempool_client_ref,
                signer=keystore.signer,
                miner_handles=mempool_handles,
                sampler_topology_hash=mempool_topology_hash,
                solver_type=MinerType.from_kind(miner_kind),
                core=core,
            )
            click.echo(
                f"  mempool handles: {[h.miner_id for h in mempool_handles]} "
                f"topology=0x{mempool_topology_hash.hex()[:16]}..."
            )

        if rest_port is not None and rest_port > 0:
            telemetry = TelemetryApiServer(
                core=core,
                client=client,
                signer=keystore.signer,
                controller=pow_controller,  # Phase 9: surface mempool stats too
                host="127.0.0.1",
                port=rest_port,
            )
            await telemetry.start()
            click.echo(f"telemetry api: http://127.0.0.1:{rest_port}/api/v1/status")

        loop = asyncio.get_running_loop()

        def _signal_shutdown() -> None:
            for c in (pow_controller, mempool_controller):
                if c is not None:
                    try:
                        c.shutdown()
                    except Exception:  # noqa: BLE001 -- signal handler must not raise
                        pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_shutdown)

        tasks: list[asyncio.Task] = []
        if pow_controller is not None:
            tasks.append(asyncio.create_task(
                pow_controller.run(), name="pow-controller"
            ))
        if mempool_controller is not None:
            tasks.append(asyncio.create_task(
                mempool_controller.run(), name="mempool-controller"
            ))

        if not tasks:
            click.echo("internal error: no controller tasks created", err=True)
            return 2

        exit_code = 0
        # Wait until any controller exits (one failing should bring down
        # the whole process -- operators can re-spawn). Once one returns,
        # signal the other to stop and wait for it to drain.
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for t in done:
            exc = t.exception()
            if exc is not None:
                click.echo(
                    f"controller {t.get_name()} exited with error:\n"
                    + "".join(traceback.format_exception(
                        type(exc), exc, exc.__traceback__
                    )),
                    err=True,
                )
                exit_code = 1
        _signal_shutdown()
        for t in pending:
            try:
                await asyncio.wait_for(t, timeout=_SHUTDOWN_GRACE_SECONDS)
            except asyncio.TimeoutError:
                click.echo(
                    f"controller {t.get_name()} did not stop within "
                    f"{_SHUTDOWN_GRACE_SECONDS}s; cancelling",
                    err=True,
                )
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception as drain_exc:
                    click.echo(
                        f"controller {t.get_name()} raised during cancel: "
                        f"{drain_exc}",
                        err=True,
                    )
            except asyncio.CancelledError:
                pass
            except Exception as drain_exc:
                click.echo(
                    f"controller {t.get_name()} raised during drain: {drain_exc}",
                    err=True,
                )

        return exit_code

    finally:
        if pow_controller is not None:
            click.echo(f"  pow stats:     {pow_controller.stats}")
        if mempool_controller is not None:
            click.echo(f"  mempool stats: {mempool_controller.stats}")
        if telemetry is not None:
            await telemetry.stop()
        if mempool_client is not None:
            await mempool_client.close()
        if client is not None:
            await client.close()
        core.close()


def _split_handles_for_mode(mode: str, handles: list) -> tuple:
    """Split `MinerHandle`s between PoW and mempool controllers.

    Returns `(pow_handles, mempool_handles)`. The split is static —
    handles assigned to PoW only ever mine PoW work, etc. Phase 9 may
    introduce dynamic re-allocation.

      - mode=pow:     (handles, [])
      - mode=mempool: ([], handles)
      - mode=both:    floor(n/2) handles to PoW, the rest to mempool.
                      For odd `n` the remainder favors mempool. For
                      n=1 this returns ([], handles) — the CLI then
                      fails fast on the both/empty-PoW combination.
    """
    mode = mode.lower()
    if mode == "pow":
        return list(handles), []
    if mode == "mempool":
        return [], list(handles)
    if mode == "both":
        n = len(handles)
        pow_count = n // 2
        return list(handles[:pow_count]), list(handles[pow_count:])
    raise ValueError(f"unknown mode {mode!r}; expected 'pow', 'mempool', or 'both'")


_MODE_HELP = (
    "Work source: pow (chain heads), mempool (QuantumComputeMempool "
    "JobProposed events), or both (split workers half-and-half; needs "
    "≥2 handles). Phase 9 will introduce a shared scheduler so one "
    "mode can use the other's idle worker."
)


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
    "--mode",
    type=click.Choice(["pow", "mempool", "both"], case_sensitive=False),
    default="pow",
    show_default=True,
    help=_MODE_HELP,
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
    "--rest-port",
    type=int,
    default=-1,
    show_default=True,
    help=_REST_PORT_HELP,
)
def quip_miner_cpu(
    node_url: str,
    signer_key_path: str,
    mode: str,
    num_cpus: int,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run CPU SA miners against a substrate chain.

    --mode controls the work source:
      pow      (default) — mine chain heads via QuantumPow.submit_proof
      mempool  — solve JobProposed orders via QuantumComputeMempool
      both     — concurrent: half the workers do PoW, the rest do mempool
                 (requires --num-cpus 2 or more)

    For mempool / both modes, the signer must be registered as a solver
    first via `quip-miner register-solver --miner-type cpu`.
    """
    import asyncio

    miner_config = {"cpu": {"num_cpus": num_cpus}}
    raise SystemExit(asyncio.run(_run_concurrent_miner(
        mode=mode,
        miner_kind="cpu",
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
    "--mode",
    type=click.Choice(["pow", "mempool", "both"], case_sensitive=False),
    default="pow",
    show_default=True,
    help=_MODE_HELP,
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=-1,
    show_default=True,
    help=_REST_PORT_HELP,
)
def quip_miner_gpu(
    node_url: str,
    signer_key_path: str,
    gpu_backend: str,
    mode: str,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run a GPU miner (CUDA / Metal / Modal) against a substrate chain.

    See `quip-miner cpu --help` for --mode semantics. GPU end-to-end
    verification against the chain is a Phase 6 follow-on; concurrent
    mode shares the same caveat about topology injection generalisation.
    """
    backend = gpu_backend.lower()
    if backend == "local":
        miner_config = {"cuda": [{"device": "0"}]}
    elif backend == "metal":
        miner_config = {"metal": [{}]}
    elif backend == "modal":
        miner_config = {"modal": [{"gpu_type": "t4"}]}
    else:
        raise click.BadParameter(f"unknown --gpu-backend: {backend}")

    raise SystemExit(asyncio.run(_run_concurrent_miner(
        mode=mode,
        miner_kind="gpu",
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
    "--mode",
    type=click.Choice(["pow", "mempool", "both"], case_sensitive=False),
    default="pow",
    show_default=True,
    help=_MODE_HELP,
)
@click.option(
    "--daily-budget",
    type=str,
    default=None,
    help="Daily QPU access-time budget. Format: '30s', '5m', '2h', '1d' (passed to QPUTimeManager)",
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=-1,
    show_default=True,
    help=_REST_PORT_HELP,
)
def quip_miner_qpu(
    node_url: str,
    signer_key_path: str,
    qpu_type: str,
    mode: str,
    daily_budget,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run a QPU miner against a substrate chain.

    Provider credentials come from the environment (e.g. DWAVE_API_KEY).
    See `quip-miner cpu --help` for --mode semantics. Same Phase 5a caveat
    as GPU: end-to-end against the chain is a Phase 6 item once topology
    binding generalises beyond CPU.
    """
    section: dict = {"type": qpu_type}
    if daily_budget is not None:
        section["daily_budget"] = daily_budget
    miner_config = {qpu_type: [section]}

    miner_kind = f"qpu_{qpu_type}" if qpu_type in ("ibm", "ionq", "pasqal") else "qpu"

    raise SystemExit(asyncio.run(_run_concurrent_miner(
        mode=mode,
        miner_kind=miner_kind,
        node_url=node_url,
        signer_key_path=signer_key_path,
        rest_port=rest_port,
        topology_spec=topology_spec,
        miner_config=miner_config,
    )))


# Entry points for console_scripts


def miner_main():
    """Entry point for the `quip-miner` console script."""
    try:
        quip_miner(standalone_mode=False)
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(f"quip-miner: error: {exc}", err=True)
        raise SystemExit(1) from exc
