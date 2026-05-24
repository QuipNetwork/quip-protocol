"""Click-based CLI for quip-miner.

Provides the `quip-miner` console command with subcommands:
- keygen     - generate a fresh sr25519 signing key
- bootstrap  - fund + register a miner account against a substrate chain
- cpu / gpu / qpu - run a miner against a substrate chain

Runtime architecture is documented in ARCHITECTURE.md at the repo root.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import signal
import socket
import ssl
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import click

from dwave_topologies.topologies.json_loader import load_topology
from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate, load
from shared.logging_config import setup_logging
from shared.mempool_miner_controller import MempoolMinerController
from shared.topology_hash import topology_hash
from shared.mempool_types import MinerType
from shared.miner_bootstrap import (
    DEFAULT_MIN_BALANCE_PLANCKS,
    BootstrapConfig,
    bootstrap,
    ensure_funded,
)
from shared.miner_config import (
    MinerConfigError,
    load_miner_config,
    merge_config,
    validate_merged,
)
from shared.miner_core import MinerCore
from substrate.client import (
    NoValidatorReachable,
    SubstrateClient,
)
from substrate.miner_controller import SubstrateMinerController
from shared.system_info import (
    DescriptorValidationError,
    build_descriptor,
    to_canonical_json,
    validate_descriptor,
)
from substrate.pool import ValidatorPool


_AUTO_IDENTIFY_LOGGER = logging.getLogger("quip_miner.auto_identify")


def _default_node_name() -> str:
    """Return the system hostname, or 'quip-miner' if unavailable.

    Used as the auto-identify fallback when neither --node-name nor TOML
    `node_name` is set. We deliberately avoid SS58-prefix fallbacks here —
    dashboards already key off the AccountId; the descriptor's node_name
    is a human-readable label and should default to something an operator
    would recognise (the rig's hostname)."""
    try:
        name = socket.gethostname()
    except OSError:
        return "quip-miner"
    return name or "quip-miner"


# Public-IP detection services queried when the operator did not set
# `public_host`. check.quip.network is the project-controlled primary;
# the rest are well-known fallbacks so detection still works when the
# primary is down. Each returns the caller's public IP in plain text.
# Order matters: we stop at the first service that returns a parseable
# IP address.
_PUBLIC_IP_SERVICES: tuple[str, ...] = (
    "https://check.quip.network",
    "https://api.ipify.org",
    "https://icanhazip.com",
    "https://ipecho.net/plain",
    "https://checkip.amazonaws.com",
    "https://ident.me",
)


async def _detect_public_ip(timeout: float = 5.0) -> Optional[str]:
    """Best-effort discovery of this host's public IP for the NodeDescriptor.

    Tries `_PUBLIC_IP_SERVICES` in order; returns the first response that
    parses as an IPv4/IPv6 address. Returns None when every service times
    out, errors, or replies with garbage. Failures are logged at debug —
    auto-identify still proceeds with `public_host=None`.

    Behaviour intentionally mirrors v0.1's `get_public_ip()` so operators
    upgrading from v0.1 see the same detected IP.
    """
    loop = asyncio.get_event_loop()
    ssl_context = ssl.create_default_context()
    for service in _PUBLIC_IP_SERVICES:
        def fetch(url: str = service) -> str:
            req = urllib.request.Request(
                url, headers={"User-Agent": "quip-miner/auto-identify"}
            )
            with urllib.request.urlopen(
                req, timeout=timeout, context=ssl_context
            ) as resp:
                return resp.read().decode("utf-8").strip()
        try:
            ip = await loop.run_in_executor(None, fetch)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            _AUTO_IDENTIFY_LOGGER.debug(
                "public IP probe failed via %s: %s", service, exc
            )
            continue
        except Exception as exc:  # noqa: BLE001 — observability path
            _AUTO_IDENTIFY_LOGGER.debug(
                "public IP probe errored via %s: %s: %s",
                service, type(exc).__name__, exc,
            )
            continue
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            _AUTO_IDENTIFY_LOGGER.debug(
                "public IP probe via %s returned non-IP %r", service, ip
            )
            continue
        _AUTO_IDENTIFY_LOGGER.info(
            "auto-identify: detected public IP %s via %s", ip, service
        )
        return ip
    _AUTO_IDENTIFY_LOGGER.warning(
        "auto-identify: could not detect public IP; descriptor will omit "
        "public_host. Set --public-host or `public_host = ...` in the "
        "miner config to skip detection."
    )
    return None


async def _auto_identify(
    client: SubstrateClient,
    keystore,
    *,
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    log_level: Optional[str],
    miners_config: dict,
) -> None:
    """Submit a signed NodeDescriptor remark using an already-connected client.

    Called once on every miner startup after the funding check. The user-
    facing contract is: identify failures never block mining — the call
    catches all exceptions and logs a warning. The descriptor's `miners`
    block is built from the same TOML-shaped dict that `MinerCore` used
    to spawn worker handles, so the descriptor always reflects the
    actual launched topology.
    """
    effective_name = (node_name or _default_node_name())[:64]
    # When the operator did not configure public_host, query
    # check.quip.network (and fallbacks) to fill it in — matches v0.1
    # behaviour. Detection is best-effort: if every service is
    # unreachable, the descriptor goes out with public_host=None.
    effective_public_host = public_host
    if effective_public_host is None:
        effective_public_host = await _detect_public_ip()
    miner_specs = _identify_specs_from_miner_config(
        effective_name, miners_config
    )
    try:
        descriptor = build_descriptor(
            node_id=effective_name,
            node_name=effective_name,
            public_host=effective_public_host,
            public_port=public_port,
            rpc_endpoints=[],
            auto_mine=True,
            log_level=(log_level or "INFO").upper(),
            miner_specs=miner_specs,
            include_system_info=True,
        )
        validate_descriptor(descriptor)
    except DescriptorValidationError as exc:
        _AUTO_IDENTIFY_LOGGER.warning(
            "auto-identify skipped: descriptor invalid (%s); "
            "mining continues",
            exc,
        )
        return
    except Exception as exc:  # noqa: BLE001 — observability path
        _AUTO_IDENTIFY_LOGGER.warning(
            "auto-identify skipped: descriptor build failed "
            "(%s: %s); mining continues",
            type(exc).__name__,
            exc,
        )
        return

    payload = to_canonical_json(descriptor)
    payload_hash = hashlib.blake2b(payload, digest_size=32).hexdigest()
    try:
        prefer_event = await client.has_call("System", "remark_with_event")
        call_function = "remark_with_event" if prefer_event else "remark"
        try:
            receipt = await client.submit_extrinsic(
                "System",
                call_function,
                {"remark": payload},
                keystore.signer,
                wait_for="inblock",
            )
        except Exception as exc:  # noqa: BLE001 — retry below or warn
            if call_function == "remark":
                raise
            _AUTO_IDENTIFY_LOGGER.warning(
                "auto-identify: remark_with_event failed (%s); "
                "retrying with plain remark",
                exc,
            )
            receipt = await client.submit_extrinsic(
                "System",
                "remark",
                {"remark": payload},
                keystore.signer,
                wait_for="inblock",
            )
            call_function = "remark"
        if receipt.error:
            _AUTO_IDENTIFY_LOGGER.warning(
                "auto-identify: %s rejected (%s); mining continues",
                call_function,
                receipt.error,
            )
            return
        _AUTO_IDENTIFY_LOGGER.info(
            "auto-identify submitted: account=%s call=System.%s "
            "extrinsic=%s payload_size=%d payload_hash=0x%s",
            keystore.signer.ss58_address(),
            call_function,
            receipt.extrinsic_hash,
            len(payload),
            payload_hash,
        )
    except Exception as exc:  # noqa: BLE001 — observability path
        _AUTO_IDENTIFY_LOGGER.warning(
            "auto-identify skipped: submission failed (%s: %s); "
            "mining continues",
            type(exc).__name__,
            exc,
        )


_VALIDATOR_HELP = (
    "Substrate validator WebSocket URL (e.g. ws://localhost:9944). "
    "Repeat to provide a failover list: tried in order at startup, "
    "and live-failover rotates through them on connection drop. "
    "Required unless provided via --config."
)
_CONFIG_HELP = (
    "Path to a TOML config file. Keys in the [miner] section serve as "
    "defaults; CLI flags override. See quip-miner.example.toml."
)
_SIGNER_KEY_HELP = "Path to the signing keystore (created by `quip-miner keygen`)"
_TOPOLOGY_HELP = (
    "Topology for the miner's sampler. Either a hardware name like "
    "'advantage2_system1' (default — the real QPU graph the chain "
    "registers) or 'zephyr:M,T' for a synthetic Zephyr Z(M,T) graph "
    "(dev / benchmark chains). Must hash to the chain's registered "
    "topology — mismatch fails fast at startup."
)
_REST_PORT_HELP = (
    "Telemetry REST API port (default 8086). Falls back to --config `rest_port`."
)
_REST_HOST_HELP = (
    "Telemetry REST API bind host. Defaults to 127.0.0.1 (loopback-only); set to "
    "0.0.0.0 to expose on all interfaces inside a container. Falls back to "
    "--config `rest_host`."
)


def _validator_option(f):
    """Repeatable `--validator URL`. See `_VALIDATOR_HELP` for semantics."""
    return click.option(
        "--validator",
        "validators",
        multiple=True,
        help=_VALIDATOR_HELP,
    )(f)


def _config_option(f):
    """Optional `--config path.toml` for TOML defaults."""
    return click.option(
        "--config",
        "config_path",
        type=click.Path(dir_okay=False),
        default=None,
        help=_CONFIG_HELP,
    )(f)


def _identification_options(f):
    """Shared identification flags applied to cpu/gpu/qpu/mempool.

    These populate the NodeDescriptor that auto-identify submits on every
    startup. All four are optional — node_name falls back to the system
    hostname, the rest are simply omitted from the descriptor when unset.
    Each one accepts a TOML override via the matching key in `[miner]`.
    """
    f = click.option(
        "--node-log",
        "node_log",
        type=click.Path(dir_okay=False),
        default=None,
        help="Path to a rotating log file (10MB x5 backups) attached "
        "alongside stderr. Falls back to --config `node_log`.",
    )(f)
    f = click.option(
        "--public-port",
        "public_port",
        type=click.IntRange(1, 65535),
        default=None,
        help="Publicly-advertised port for this miner (1-65535). "
        "Embedded in the NodeDescriptor for dashboards; does not bind. "
        "Falls back to --config `public_port`.",
    )(f)
    f = click.option(
        "--public-host",
        "public_host",
        default=None,
        help="Publicly-advertised hostname for this miner (≤253 bytes). "
        "Embedded in the NodeDescriptor; does not bind. Falls back to "
        "--config `public_host`, then auto-detection via "
        "check.quip.network (with ipify/icanhazip fallbacks).",
    )(f)
    f = click.option(
        "--node-name",
        "node_name",
        default=None,
        help="Human-readable label for this miner (max 64 UTF-8 bytes). "
        "Defaults to the system hostname. Falls back to --config `node_name`.",
    )(f)
    return f


def _faucet_url_option(f):
    """Optional `--faucet-url URL` for auto-topup on underfunded wallets.

    When set, Guard C tops up the wallet through the configured faucet
    bot before refusing to start. When unset, an underfunded wallet
    fails fast with `wallet-underfunded`.
    """
    return click.option(
        "--faucet-url",
        default=None,
        help="If set, request funding from this faucet bot when balance "
        "is below the registration threshold. Without --faucet-url, "
        "underfunded wallets fail fast with `wallet-underfunded`.",
    )(f)


def _resolve_runtime_config(
    *,
    config_path: Optional[str],
    cli_kwargs: dict,
    defaults: Optional[dict] = None,
) -> dict:
    """Merge TOML config (if any) with CLI kwargs and run validate_merged.

    Precedence: CLI kwargs > TOML > `defaults`. Defaults apply *after*
    merge so a default-valued CLI arg never shadows a TOML override
    (e.g. ~/.quip-miner/signing.json as the CLI fallback must not win
    against a TOML `signer_key = "/tmp/k.json"`).

    Returns the merged dict on success. Raises `click.ClickException` (so
    `miner_main`'s wrapper formats it as `quip-miner: error: ...`) when
    the TOML can't be loaded or required keys (`validators`, `signer_key`)
    are missing.
    """
    if config_path is not None:
        try:
            toml_data = load_miner_config(Path(config_path).expanduser())
        except MinerConfigError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        toml_data = {}
    merged = merge_config(toml_data, cli_kwargs)
    if defaults:
        for key, value in defaults.items():
            # `is None` (not truthy) so an explicit falsy override from
            # TOML — `rest_port = 0`, `auto_mine = false`, `""` — wins
            # over the default. `validate_merged` rejects empty signer_key
            # downstream, so the only "missing" state we paper over here
            # is genuine absence.
            if merged.get(key) is None:
                merged[key] = value
    try:
        validate_merged(merged)
    except MinerConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    return merged


# Fail-fast startup guards. Each helper renders a single-line, machine-
# parseable error (kebab-case code + key=value pairs) and raises
# `click.ClickException`; `miner_main` formats it as
# `quip-miner: error: <code> <kv>...` on stderr and exits 1.
#
# Codes:
#   wallet-not-configured     — keystore path doesn't exist
#   wallet-load-failed        — keystore exists but couldn't be parsed
#   validators-unreachable    — every URL in the rotation refused connect
#   wallet-underfunded        — balance below threshold, no faucet wired
#   wallet-faucet-failed      — faucet was configured but the top-up failed


def _load_keystore_or_fail(path_str: str):
    """Guard A — wallet configured.

    Returns the loaded hybrid keystore. Raises ClickException with
    `wallet-not-configured` (missing file) or `wallet-load-failed`
    (existed but unreadable / wrong schema).
    """
    path = Path(path_str).expanduser()
    if not path.exists():
        raise click.ClickException(f"wallet-not-configured keystore={path}")
    try:
        return load(path)
    except Exception as exc:  # noqa: BLE001 — surfaced via exception code
        raise click.ClickException(
            f"wallet-load-failed keystore={path} error={type(exc).__name__}"
        ) from exc


async def _connect_or_fail(urls: tuple[str, ...]) -> SubstrateClient:
    """Guard B — validators reachable.

    Constructs a direct ``SubstrateClient`` over the configured URL list
    (with its own walk-on-connect failover). One-shot CLI subcommands
    (register, deregister, identify, …) don't run long enough to benefit
    from the pool's hot-active swap, so they skip the pool entirely.

    On ``NoValidatorReachable``, renders the structured attempt log as
    ``urls=<csv> reasons=<csv>`` and raises ``ClickException`` with
    ``validators-unreachable``.
    """
    client = SubstrateClient(urls=urls)
    try:
        await client.connect()
    except NoValidatorReachable as exc:
        url_csv = ",".join(a.url for a in exc.attempts)
        reasons = ",".join(a.exc_type for a in exc.attempts)
        raise click.ClickException(
            f"validators-unreachable urls={url_csv} reasons={reasons}"
        ) from exc
    return client


async def _ensure_funded_or_fail(
    client: SubstrateClient,
    keystore,
    *,
    faucet_url: Optional[str],
    min_balance: int,
) -> int:
    """Guard C — wallet funded.

    If balance is already at or above threshold, returns the balance.
    When `faucet_url` is set, reuses `shared.miner_bootstrap._ensure_funded`
    to top up via the configured faucet bot. Underfunded with no faucet
    raises `wallet-underfunded`; faucet failure raises
    `wallet-faucet-failed`.
    """
    account = keystore.signer.account_id_bytes()
    balance = await client.query_balance(account)
    if balance >= min_balance:
        return balance
    if not faucet_url:
        raise click.ClickException(
            f"wallet-underfunded ss58={keystore.signer.ss58_address()} "
            f"balance={balance} threshold={min_balance}"
        )
    # Construct a minimal BootstrapConfig just to drive the existing
    # faucet helper — its `_post_faucet` + settlement-poll behavior is
    # exactly what we want here.
    cfg = BootstrapConfig(
        validators=client.urls,
        signer_key_path=keystore.path,
        faucet_url=faucet_url,
        min_balance_plancks=min_balance,
    )
    try:
        return await ensure_funded(client, keystore, cfg)
    except NoValidatorReachable as exc:
        # Faucet path stays connected via SubstrateClient; if the rotation
        # collapses mid-settlement, surface it as the same code the connect
        # guard uses rather than burying it in `wallet-faucet-failed`.
        urls = ",".join(a.url for a in exc.attempts)
        reasons = ",".join(a.exc_type for a in exc.attempts)
        raise click.ClickException(
            f"validators-unreachable urls={urls} reasons={reasons}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — translated to a CLI error code
        raise click.ClickException(
            f"wallet-faucet-failed ss58={keystore.signer.ss58_address()} "
            f"balance={balance} threshold={min_balance} "
            f"error={type(exc).__name__}"
        ) from exc


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
@_validator_option
@_config_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to the keystore (will be created if missing). "
    "Falls back to --config `signer_key`, then ~/.quip-miner/signing.json.",
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
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
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

    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={
            "validators": validators,
            "signer_key": signer_key_path,
            "faucet_url": faucet_url,
        },
        defaults={"signer_key": "~/.quip-miner/signing.json"},
    )

    config = BootstrapConfig(
        validators=tuple(merged["validators"]),
        signer_key_path=Path(merged["signer_key"]).expanduser(),
        faucet_url=merged.get("faucet_url"),
        sudo_key_uri=sudo_key_uri,
        seed_chain=seed_chain,
        seed_topology_mt=topology_mt,
    )
    # bootstrap() does its own connect + balance check; translate the two
    # failures the operator can fix into the same machine-parseable codes
    # the mining commands use.
    try:
        result = asyncio.run(bootstrap(config))
    except NoValidatorReachable as exc:
        urls = ",".join(a.url for a in exc.attempts)
        reasons = ",".join(a.exc_type for a in exc.attempts)
        raise click.ClickException(
            f"validators-unreachable urls={urls} reasons={reasons}"
        ) from exc

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
@_validator_option
@_config_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False, exists=True),
    default=None,
    help="Path to the signing keystore (run `quip-miner keygen` first). "
    "Falls back to --config `signer_key`.",
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
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
    miner_type: str,
) -> None:
    """Register the keystore's account as a QuantumComputeMempool solver.

    One-time setup (idempotent if already registered). The solver type is
    surfaced to job proposers via `mode = Bid{miner_types: [...]}` filters.
    Use `quip-miner deregister-solver` to opt out.
    """
    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={"validators": validators, "signer_key": signer_key_path},
        defaults={"signer_key": "~/.quip-miner/signing.json"},
    )
    keystore = _load_keystore_or_fail(merged["signer_key"])
    mt = MinerType.from_kind(miner_type)

    async def _do() -> int:
        # One-shot CLI: skip the pool — direct client over the URL list
        # is enough.
        client = await _connect_or_fail(tuple(merged["validators"]))
        try:
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
@_validator_option
@_config_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False, exists=True),
    default=None,
    help="Path to the signing keystore. Falls back to --config `signer_key`.",
)
def quip_miner_deregister_solver(
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
) -> None:
    """Deregister the keystore's solver from QuantumComputeMempool.

    Idempotent against an unregistered account (returns 0 with a no-op
    message). After deregistration, submit_solution / claim_reward
    extrinsics will fail with `SolverNotRegistered` until you re-register.
    """
    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={"validators": validators, "signer_key": signer_key_path},
        defaults={"signer_key": "~/.quip-miner/signing.json"},
    )
    keystore = _load_keystore_or_fail(merged["signer_key"])

    async def _do() -> int:
        client = await _connect_or_fail(tuple(merged["validators"]))
        try:
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
# Topology binding is explicit via --topology. The default is the
# Advantage2_system1 hardware topology (the project-wide DEFAULT_TOPOLOGY),
# which matches what the chain registers. The synthetic 'zephyr:M,T' form
# stays available for benchmarks and dev chains seeded with a perfect
# Zephyr graph. The CLI verifies at startup that the sampler's topology
# hash matches the chain's snapshot; mismatch fails fast with the
# registered hash printed so the operator can fix --topology or re-seed
# the chain.
# --------------------------------------------------------------------------


def _parse_topology(spec: str):
    """Parse a topology spec into a sampler-compatible topology object.

    Accepted forms:
      - `advantage2_system1` (and other hardware names handled by
        `load_topology`) — real QPU hardware graph from the bundled JSON.
      - `zephyr:M,T` — perfect Zephyr Z(M,T) generated on the fly.

    Returns the topology so the CLI can plug it straight into the spec's
    `args["topology"]`. Hardware names are case-insensitive and accept
    either `_` or `-` separators.
    """
    if ":" in spec:
        family, params = spec.split(":", 1)
        if family.lower() != "zephyr":
            raise click.BadParameter(
                f"only 'zephyr:M,T' is supported for synthetic topologies "
                f"(got {family!r}); for hardware use a bare name like "
                f"'advantage2_system1'"
            )
        try:
            m_str, t_str = params.split(",")
            m, t = int(m_str), int(t_str)
        except ValueError as exc:
            raise click.BadParameter(
                f"--topology zephyr params must be 'M,T' (got {params!r})"
            ) from exc
        return zephyr(m, t)

    # Bare name → delegate to load_topology (handles Advantage2_system1,
    # named constants, file paths). load_topology raises ValueError /
    # FileNotFoundError; translate to click.BadParameter for a clean CLI
    # error rather than a stack trace.
    try:
        return load_topology(spec)
    except (ValueError, FileNotFoundError) as exc:
        raise click.BadParameter(
            f"--topology {spec!r}: {exc}"
        ) from exc


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


# Seconds to wait for a controller to drain after signalling shutdown.
# Must exceed the longest controller poll interval (<=10s for head-subscription
# timeout in SubstrateMinerController).
_SHUTDOWN_GRACE_SECONDS = 15.0


async def _run_concurrent_miner(
    *,
    mode: str,
    miner_kind: str,
    validators: tuple,
    signer_key_path: str,
    faucet_url: Optional[str],
    rest_port: int,
    rest_host: str,
    topology_spec: str,
    miner_config: dict,
    node_name: Optional[str] = None,
    public_host: Optional[str] = None,
    public_port: Optional[int] = None,
    node_log: Optional[str] = None,
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

    # node_log is opt-in (TOML or CLI). When set, re-run setup_logging
    # so a RotatingFileHandler is attached alongside the stderr handler
    # that the click group installed. The current root level is preserved.
    if node_log:
        current_level = logging.getLevelName(logging.getLogger().level)
        setup_logging(
            log_level=current_level,
            node_log_file=node_log,
            node_name=node_name or _default_node_name(),
        )

    # Guard A — wallet configured (keystore exists + loads).
    keystore = _load_keystore_or_fail(signer_key_path)
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
    pool: Optional[ValidatorPool] = None
    pow_controller = None
    mempool_controller = None

    setup_client: Optional[SubstrateClient] = None
    try:
        pool = ValidatorPool(urls=tuple(validators))
        # Guard B — validators reachable. The setup client is a direct
        # SubstrateClient used for the startup sequence (Guards B/C,
        # auto-identify, initial topology-snapshot read). It's closed
        # before controllers start; the controllers each own their own
        # in-parent build_client and route reads + submissions through
        # the swap-aware pool.
        client = setup_client = await _connect_or_fail(tuple(validators))
        # Guard C — wallet funded (with optional faucet top-up).
        await _ensure_funded_or_fail(
            client,
            keystore,
            faucet_url=faucet_url,
            min_balance=DEFAULT_MIN_BALANCE_PLANCKS,
        )

        # Auto-identify: publish a signed NodeDescriptor remark so
        # dashboards can map our AccountId → node_name + advertised
        # hardware. Runs on every startup; failures log a warning and
        # never block mining. There is no opt-out — dashboard visibility
        # is part of the miner contract.
        await _auto_identify(
            client,
            keystore,
            node_name=node_name,
            public_host=public_host,
            public_port=public_port,
            log_level=None,
            miners_config=miner_config,
        )

        # Both PoW and mempool need the chain's registered allowed-value
        # specs to derive a topology hash that matches the chain's
        # canonical `hash_topology`. Pull the snapshot once.
        snapshot = None
        if pow_handles or mempool_handles:
            try:
                head = await client.get_head()
                snapshot = await client.get_mining_snapshot(
                    at=head,
                    miner_account_bytes=keystore.signer.account_id_bytes(),
                )
            except Exception as exc:
                click.echo(
                    f"failed to query chain state: {exc}", err=True
                )
                return 3
            if snapshot is None:
                click.echo(
                    "chain has no registered topology; run "
                    "`quip-miner bootstrap --seed-chain` first",
                    err=True,
                )
                return 3

        # Resolve telemetry port. The sibling process is unconditional;
        # any non-positive `rest_port` (legacy `-1` default, `0`, `None`)
        # collapses to 8086.
        telemetry_port = rest_port if rest_port and rest_port > 0 else 8086

        if pow_handles:
            # PoW requires the sampler's topology to match the chain's
            # registered DefaultTopology (the chain validates this in
            # `submit_proof` via `InvalidTopology`).
            expected_hash = topology_hash(
                topology.nodes,
                topology.edges,
                snapshot.allowed_h_values,
                snapshot.allowed_j_values,
                snapshot.allowed_spin_values,
            )
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
                pool=pool,
                signer=keystore.signer,
                miner_handles=pow_handles,
                topology_hash=pow_topology_hash,
                core=core,
                telemetry_port=telemetry_port,
            )
            click.echo(
                f"  pow handles: {[h.miner_id for h in pow_handles]} "
                f"topology=0x{pow_topology_hash.hex()[:16]}..."
            )

        if mempool_handles:
            # Bind the mempool sampler hash to the same canonical form
            # the chain uses (nodes, edges, allowed_h, allowed_j,
            # allowed_spin). The allowed-value specs come from the chain
            # snapshot so client and chain stay in lockstep.
            mempool_topology_hash = topology_hash(
                topology.nodes,
                topology.edges,
                snapshot.allowed_h_values,
                snapshot.allowed_j_values,
                snapshot.allowed_spin_values,
            )
            mempool_controller = MempoolMinerController(
                pool=pool,
                signer=keystore.signer,
                miner_handles=mempool_handles,
                sampler_topology_hash=mempool_topology_hash,
                allowed_h_values=snapshot.allowed_h_values,
                allowed_j_values=snapshot.allowed_j_values,
                allowed_spin_values=snapshot.allowed_spin_values,
                solver_type=MinerType.from_kind(miner_kind),
                core=core,
            )
            click.echo(
                f"  mempool handles: {[h.miner_id for h in mempool_handles]} "
                f"topology=0x{mempool_topology_hash.hex()[:16]}..."
            )

        click.echo(
            f"telemetry api: http://{rest_host}:{telemetry_port}/api/v1/status "
            "(sibling process)"
        )

        # Setup done — close the direct client so controllers own the
        # only live parent-side connections from here on.
        await setup_client.close()
        setup_client = None

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
        if setup_client is not None:
            # Setup failed before the controllers took over; clean up
            # the standalone connection.
            await setup_client.close()
        if pool is not None:
            await pool.close()
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
@_validator_option
@_config_option
@_faucet_url_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default=None,
    help=_SIGNER_KEY_HELP + " Falls back to --config `signer_key`, "
    "then ~/.quip-miner/signing.json.",
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
    default="advantage2_system1",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=None,
    show_default=False,
    help=_REST_PORT_HELP,
)
@click.option(
    "--rest-host",
    type=str,
    default=None,
    show_default=False,
    help=_REST_HOST_HELP,
)
@_identification_options
def quip_miner_cpu(
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
    faucet_url: Optional[str],
    mode: str,
    num_cpus: int,
    topology_spec: str,
    rest_port: Optional[int],
    rest_host: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    node_log: Optional[str],
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
    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={
            "validators": validators,
            "signer_key": signer_key_path,
            "faucet_url": faucet_url,
            "rest_port": rest_port,
            "rest_host": rest_host,
            "node_name": node_name,
            "public_host": public_host,
            "public_port": public_port,
            "node_log": node_log,
        },
        defaults={
            "signer_key": "~/.quip-miner/signing.json",
            "rest_port": -1,
            "rest_host": "127.0.0.1",
        },
    )
    miner_config = {"cpu": {"num_cpus": num_cpus}}
    raise SystemExit(asyncio.run(_run_concurrent_miner(
        mode=mode,
        miner_kind="cpu",
        validators=tuple(merged["validators"]),
        signer_key_path=merged["signer_key"],
        faucet_url=merged.get("faucet_url"),
        rest_port=int(merged["rest_port"]),
        rest_host=str(merged["rest_host"]),
        topology_spec=topology_spec,
        miner_config=miner_config,
        node_name=merged.get("node_name"),
        public_host=merged.get("public_host"),
        public_port=merged.get("public_port"),
        node_log=merged.get("node_log"),
    )))


@quip_miner.command("gpu")
@_validator_option
@_config_option
@_faucet_url_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default=None,
    help=_SIGNER_KEY_HELP + " Falls back to --config `signer_key`, "
    "then ~/.quip-miner/signing.json.",
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
    default="advantage2_system1",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=None,
    show_default=False,
    help=_REST_PORT_HELP,
)
@click.option(
    "--rest-host",
    type=str,
    default=None,
    show_default=False,
    help=_REST_HOST_HELP,
)
@_identification_options
def quip_miner_gpu(
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
    faucet_url: Optional[str],
    gpu_backend: str,
    mode: str,
    topology_spec: str,
    rest_port: Optional[int],
    rest_host: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    node_log: Optional[str],
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

    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={
            "validators": validators,
            "signer_key": signer_key_path,
            "faucet_url": faucet_url,
            "rest_port": rest_port,
            "rest_host": rest_host,
            "node_name": node_name,
            "public_host": public_host,
            "public_port": public_port,
            "node_log": node_log,
        },
        defaults={
            "signer_key": "~/.quip-miner/signing.json",
            "rest_port": -1,
            "rest_host": "127.0.0.1",
        },
    )
    raise SystemExit(asyncio.run(_run_concurrent_miner(
        mode=mode,
        miner_kind="gpu",
        validators=tuple(merged["validators"]),
        signer_key_path=merged["signer_key"],
        faucet_url=merged.get("faucet_url"),
        rest_port=int(merged["rest_port"]),
        rest_host=str(merged["rest_host"]),
        topology_spec=topology_spec,
        miner_config=miner_config,
        node_name=merged.get("node_name"),
        public_host=merged.get("public_host"),
        public_port=merged.get("public_port"),
        node_log=merged.get("node_log"),
    )))


@quip_miner.command("qpu")
@_validator_option
@_config_option
@_faucet_url_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default=None,
    help=_SIGNER_KEY_HELP + " Falls back to --config `signer_key`, "
    "then ~/.quip-miner/signing.json.",
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
    default="advantage2_system1",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=None,
    show_default=False,
    help=_REST_PORT_HELP,
)
@click.option(
    "--rest-host",
    type=str,
    default=None,
    show_default=False,
    help=_REST_HOST_HELP,
)
@_identification_options
def quip_miner_qpu(
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
    faucet_url: Optional[str],
    qpu_type: str,
    mode: str,
    daily_budget: Optional[str],
    topology_spec: str,
    rest_port: Optional[int],
    rest_host: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    node_log: Optional[str],
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

    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={
            "validators": validators,
            "signer_key": signer_key_path,
            "faucet_url": faucet_url,
            "rest_port": rest_port,
            "rest_host": rest_host,
            "node_name": node_name,
            "public_host": public_host,
            "public_port": public_port,
            "node_log": node_log,
        },
        defaults={
            "signer_key": "~/.quip-miner/signing.json",
            "rest_port": -1,
            "rest_host": "127.0.0.1",
        },
    )
    raise SystemExit(asyncio.run(_run_concurrent_miner(
        mode=mode,
        miner_kind=miner_kind,
        validators=tuple(merged["validators"]),
        signer_key_path=merged["signer_key"],
        faucet_url=merged.get("faucet_url"),
        rest_port=int(merged["rest_port"]),
        rest_host=str(merged["rest_host"]),
        topology_spec=topology_spec,
        miner_config=miner_config,
        node_name=merged.get("node_name"),
        public_host=merged.get("public_host"),
        public_port=merged.get("public_port"),
        node_log=merged.get("node_log"),
    )))


# --------------------------------------------------------------------------
# `quip-miner identify` — post a signed NodeDescriptor remark.
#
# Builds a v0.1-shaped `nodes.json` entry (schema=quip.node_descriptor.v1)
# from auto-detected hardware + optional --miner-config, canonicalizes to
# UTF-8 JSON, and submits via System.remark_with_event (falling back to
# System.remark if the runtime metadata doesn't expose the eventful
# variant). The signer's account is the canonical identity; the dashboard
# indexer maps AccountId -> descriptor by scanning these remarks. See
# MINERSURVEY.md for the wire format and Phase 2 roadmap.
# --------------------------------------------------------------------------


def _load_identify_miners_toml(path: Path) -> dict:
    """Read a TOML file shaped like `MinerCore`'s `miners_config` arg.

    Unlike `load_miner_config` (which only returns the `[miner]` table),
    this returns the top-level `cpu` / `gpu` / `qpu` / device sections
    that `MinerCore._initialize_miners` consumes. `identify` uses this
    so operators can describe their miner inventory in TOML form:

        [cpu]
        num_cpus = 2

        [gpu]
        backend = "cuda"
        devices = "0,1"

        [dwave]
        solver = "Advantage2_system1"
        daily_budget = "5m"

    The `[miner]` section, if present, is ignored — it carries
    connection/keystore config that's already supplied via CLI flags.
    """
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    if not path.exists():
        raise click.ClickException(f"miner config not found: {path}")
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise click.ClickException(
            f"miner config parse failed ({path}): {exc}"
        ) from exc
    raw.pop("miner", None)
    return raw


def _identify_specs_from_miner_config(
    node_id: str, miners_config: dict
) -> list:
    """Materialize miner spec dicts from a parsed miner-config TOML.

    Mirrors `MinerCore._initialize_miners` but produces plain dicts
    instead of spawning `MinerHandle` worker processes — the identify
    flow only needs the spec shape to populate the `miners` block of
    the descriptor.
    """
    from shared.miner_core import (
        _GPU_DEVICE_SECTIONS,
        _QPU_DEVICE_SECTIONS,
        _build_gpu_specs,
        _build_qpu_specs,
    )

    specs: list = []
    cpu_cfg = miners_config.get("cpu")
    if cpu_cfg is not None:
        num_cpus = int(cpu_cfg.get("num_cpus", 1))
        cpu_args = dict(cpu_cfg.get("args", {}))
        for i in range(num_cpus):
            specs.append({
                "id": f"{node_id}-CPU-{i + 1}",
                "kind": "cpu",
                "args": cpu_args,
            })
    has_gpu = miners_config.get("gpu") is not None or any(
        miners_config.get(k) is not None for k in _GPU_DEVICE_SECTIONS
    )
    if has_gpu:
        specs.extend(_build_gpu_specs(node_id, miners_config))
    has_qpu = miners_config.get("qpu") is not None or any(
        miners_config.get(k) is not None for k in _QPU_DEVICE_SECTIONS
    )
    if has_qpu:
        specs.extend(_build_qpu_specs(node_id, miners_config))
    return specs


@quip_miner.command("identify")
@_validator_option
@_config_option
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to the signing keystore. Falls back to --config `signer_key`, "
    "then ~/.quip-miner/signing.json.",
)
@click.option(
    "--node-name",
    required=True,
    help="Human-readable label for this miner (max 64 UTF-8 bytes). "
    "Dashboards display this; awards still key off AccountId.",
)
@click.option(
    "--rpc-endpoint",
    "rpc_endpoints",
    multiple=True,
    help="Publicly-advertised RPC endpoint for this miner. Repeat for up "
    "to 8 entries. Distinct from --validator: that's where we submit, "
    "this is what dashboards publish.",
)
@click.option("--public-host", default=None, help="Public hostname (≤253 bytes)")
@click.option(
    "--public-port",
    type=click.IntRange(1, 65535),
    default=None,
    help="Public port (1-65535)",
)
@click.option(
    "--auto-mine/--no-auto-mine",
    "auto_mine",
    default=False,
    help="Mark this node as auto-mining on startup.",
)
@click.option(
    "--descriptor-log-level",
    "descriptor_log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="log_level value to embed in the descriptor (separate from "
    "the CLI's own --log-level which controls quip-miner output).",
)
@click.option(
    "--miner-config",
    "miner_config_path",
    type=click.Path(dir_okay=False, exists=True),
    default=None,
    help="Path to a quip-miner.toml whose [cpu]/[gpu]/[qpu]/[cuda]/[metal]/"
    "[modal]/[dwave] sections describe this node's miner inventory. "
    "Populates the descriptor's `miners` block.",
)
@click.option(
    "--no-system-info",
    "skip_system_info",
    is_flag=True,
    default=False,
    help="Skip hardware probe (omits `system_info`). Useful in CI / "
    "sandboxed environments where subprocess calls aren't permitted.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the canonical JSON payload to stdout and exit without "
    "submitting. Useful for previewing exactly what would land on chain.",
)
def quip_miner_identify(
    validators: tuple,
    config_path: Optional[str],
    signer_key_path: Optional[str],
    node_name: str,
    rpc_endpoints: tuple,
    public_host: Optional[str],
    public_port: Optional[int],
    auto_mine: bool,
    descriptor_log_level: str,
    miner_config_path: Optional[str],
    skip_system_info: bool,
    dry_run: bool,
) -> None:
    """Post a signed NodeDescriptor remark for this miner.

    The descriptor carries node_name, optional public_host/public_port,
    advertised rpc_endpoints, auto_mine flag, runtime block (python /
    quip_version / docker), miners block (from --miner-config), and an
    auto-detected system_info block (skippable with --no-system-info).
    Submitted as `System.remark_with_event` so block indexers see the
    canonical JSON without needing to scan extrinsic call data.
    """
    merged = _resolve_runtime_config(
        config_path=config_path,
        cli_kwargs={
            "validators": validators,
            "signer_key": signer_key_path,
        },
        defaults={"signer_key": "~/.quip-miner/signing.json"},
    ) if not dry_run else {
        # In --dry-run we don't need validators or a connected client —
        # only the signer (for the account id printed in the preview).
        "signer_key": signer_key_path or "~/.quip-miner/signing.json",
        "validators": tuple(validators),
    }

    keystore = _load_keystore_or_fail(merged["signer_key"])
    node_id = node_name

    miners_config: dict = {}
    if miner_config_path is not None:
        miners_config = _load_identify_miners_toml(
            Path(miner_config_path).expanduser()
        )
    miner_specs = _identify_specs_from_miner_config(node_id, miners_config)

    descriptor = build_descriptor(
        node_id=node_id,
        node_name=node_name,
        public_host=public_host,
        public_port=public_port,
        rpc_endpoints=list(rpc_endpoints),
        auto_mine=auto_mine,
        log_level=descriptor_log_level.upper(),
        miner_specs=miner_specs,
        include_system_info=not skip_system_info,
    )

    try:
        validate_descriptor(descriptor)
    except DescriptorValidationError as exc:
        raise click.ClickException(f"descriptor-invalid {exc}") from exc

    payload = to_canonical_json(descriptor)
    payload_hash = hashlib.blake2b(payload, digest_size=32).hexdigest()

    if dry_run:
        click.echo(payload.decode("utf-8"))
        click.echo(
            f"\n# account            : {keystore.signer.ss58_address()}", err=True
        )
        click.echo(f"# payload_size_bytes : {len(payload)}", err=True)
        click.echo(f"# payload_hash       : 0x{payload_hash}", err=True)
        return

    async def _do() -> int:
        client = await _connect_or_fail(tuple(merged["validators"]))
        try:
            prefer_event = await client.has_call("System", "remark_with_event")
            call_function = "remark_with_event" if prefer_event else "remark"
            try:
                receipt = await client.submit_extrinsic(
                    "System",
                    call_function,
                    {"remark": payload},
                    keystore.signer,
                    wait_for="inblock",
                )
            except Exception as exc:  # noqa: BLE001 — surfaced via CLI code
                if call_function == "remark":
                    raise
                # Some metadata caches answer "yes" but the active
                # runtime rejects the call at compose time — degrade to
                # plain remark rather than failing the whole identify.
                click.echo(
                    f"remark_with_event submission failed ({exc}); "
                    f"retrying with plain remark",
                    err=True,
                )
                receipt = await client.submit_extrinsic(
                    "System",
                    "remark",
                    {"remark": payload},
                    keystore.signer,
                    wait_for="inblock",
                )
                call_function = "remark"
            if receipt.error:
                click.echo(f"{call_function} failed: {receipt.error}", err=True)
                return 3
            click.echo("identify submitted")
            click.echo(f"  account            : {keystore.signer.ss58_address()}")
            click.echo(f"  call               : System.{call_function}")
            click.echo(f"  extrinsic_hash     : {receipt.extrinsic_hash}")
            click.echo(f"  block_hash         : {receipt.block_hash}")
            click.echo(f"  payload_size_bytes : {len(payload)}")
            click.echo(f"  payload_hash       : 0x{payload_hash}")
            return 0
        finally:
            await client.close()

    raise SystemExit(asyncio.run(_do()))


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
