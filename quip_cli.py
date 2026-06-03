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
import os
import signal
import socket
import ssl
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import click

from dwave_topologies.topologies.json_loader import load_topology
from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate, load
from shared.logging_config import setup_logging
from shared.mempool_miner_controller import MempoolMinerController
from shared.mempool_types import MinerType, qpu_miner_kind
from shared.miner_bootstrap import (
    DEFAULT_MIN_BALANCE_PLANCKS,
    BootstrapConfig,
    Underfunded,
    _ensure_registered,
    bootstrap,
    ensure_funded_via_faucet,
)
from shared.miner_config import (
    GPU_BACKEND_SECTIONS,
    MODE_NAMES,
    MinerConfigError,
    ModeResolutionError,
    QPU_BACKEND_SECTIONS,
    SubmissionConfig,
    load_backend_config,
    load_miner_config,
    load_submission_config,
    load_toml,
    merge_config,
    present_backend_groups,
    resolve_mode,
    resolve_modes,
    validate_merged,
)
from shared.miner_core import MinerCore, build_miner_specs
from substrate.client import (
    NoRegisteredTopology,
    NoValidatorReachable,
    SubstrateClient,
)
from substrate.miner_controller import SubstrateMinerController
from substrate.remark import submit_remark
from shared.system_info import (
    DescriptorValidationError,
    _scrub,
    build_descriptor,
    to_canonical_json,
    validate_descriptor,
)
from shared.version import PROTOCOL_VERSION, get_version
from substrate.pool import ValidatorPool


_AUTO_IDENTIFY_LOGGER = logging.getLogger("quip_miner.auto_identify")
_STARTUP_LOGGER = logging.getLogger("quip_miner.startup")

# Funding, registration, and descriptor filing must each succeed before the
# miner runs — they retry this many times with exponential backoff (delays
# 5,10,20,40,45,45,45s ≈ 3.5 minutes) before failing the startup hard. The
# node descriptor is filed right after register_miner, so its remark can race
# the registration extrinsic's nonce (stale `accountNextIndex` for ~a block);
# the retry rides that out. The per-round "participating" remark is separate
# and stays best-effort — it is not gated here.
_STARTUP_RETRY_ATTEMPTS = 8
_STARTUP_RETRY_BASE_DELAY_SECONDS = 5.0
_STARTUP_RETRY_MAX_DELAY_SECONDS = 45.0


async def _retry_until_verified(label: str, attempt) -> str:
    """Run async ``attempt() -> (ok, detail)`` until it verifies or gives up.

    Retries up to ``_STARTUP_RETRY_ATTEMPTS`` times with exponential backoff,
    logging loudly at WARNING between tries. Returns the success ``detail`` on
    the first verified attempt; raises ``RuntimeError(detail)`` once attempts
    are exhausted so the caller can translate it into a fatal CLI error code.
    """
    delay = _STARTUP_RETRY_BASE_DELAY_SECONDS
    detail = "no response"
    for i in range(1, _STARTUP_RETRY_ATTEMPTS + 1):
        ok, detail = await attempt()
        if ok:
            return detail
        if i < _STARTUP_RETRY_ATTEMPTS:
            _STARTUP_LOGGER.warning(
                "%s: attempt %d/%d NOT verified (%s); retrying in %.0fs",
                label, i, _STARTUP_RETRY_ATTEMPTS, detail, delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, _STARTUP_RETRY_MAX_DELAY_SECONDS)
    raise RuntimeError(detail)


def _validators_unreachable(exc: NoValidatorReachable) -> click.ClickException:
    """Build a ``ClickException`` for a ``NoValidatorReachable`` failure.

    Renders the structured attempt log as ``urls=<csv> reasons=<csv>``.
    Callers should ``raise _validators_unreachable(exc) from exc``.
    """
    urls = ",".join(a.url for a in exc.attempts)
    reasons = ",".join(a.exc_type for a in exc.attempts)
    return click.ClickException(f"validators-unreachable urls={urls} reasons={reasons}")


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
    """Guard E — file a signed NodeDescriptor remark and verify it lands.

    Called on every startup after registration. Filing the descriptor is a
    fatal startup requirement: a descriptor that can't be built/validated
    fails immediately (operator misconfiguration), and submission is retried
    over several minutes before failing hard via the ``descriptor-failed`` CLI
    code. The per-round "participating" remark is a separate, best-effort
    signal and is not gated here. The descriptor's `miners` block is built
    from the same TOML-shaped dict that `MinerCore` used to spawn worker
    handles, so the descriptor always reflects the actual launched topology.
    """
    effective_name = (node_name or _default_node_name())[:64]
    # When the operator did not configure public_host, query
    # check.quip.network (and fallbacks) to fill it in — matches v0.1
    # behaviour. Detection is best-effort: if every service is
    # unreachable, the descriptor goes out with public_host=None.
    effective_public_host = public_host
    if effective_public_host is None:
        effective_public_host = await _detect_public_ip()
    miner_specs = build_miner_specs(effective_name, miners_config)
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
        # Deterministic: retrying won't fix an invalid descriptor. Fail hard.
        raise click.ClickException(
            f"descriptor-invalid ss58={keystore.signer.ss58_address()} "
            f"error={exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — translated to a fatal CLI code
        raise click.ClickException(
            f"descriptor-build-failed ss58={keystore.signer.ss58_address()} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc

    payload = to_canonical_json(descriptor)
    payload_hash = hashlib.blake2b(payload, digest_size=32).hexdigest()

    # Retry submission over several minutes: a remark fired right after
    # register_miner can be rejected on a stale nonce until the registration
    # block is imported. Filing the descriptor is a fatal startup requirement.
    async def _attempt():
        return await _submit_descriptor_remark(
            client, keystore, payload=payload, payload_hash=payload_hash
        )

    try:
        await _retry_until_verified("file-descriptor", _attempt)
    except RuntimeError as exc:
        raise click.ClickException(
            f"descriptor-failed ss58={keystore.signer.ss58_address()} "
            f"error={exc}"
        ) from exc


async def _submit_descriptor_remark(
    client: SubstrateClient,
    keystore,
    *,
    payload: bytes,
    payload_hash: str,
) -> Tuple[bool, str]:
    """Submit the descriptor remark once.

    Returns ``(True, "")`` when it lands in-block, else ``(False, detail)``
    so the caller can back off and retry. Prefers
    ``System.remark_with_event`` (dashboards key off the event) and falls back
    to plain ``remark`` if the event variant fails to compose against the live
    runtime.
    """
    def _warn_fallback(exc: Exception) -> None:
        _AUTO_IDENTIFY_LOGGER.warning(
            "auto-identify: remark_with_event failed (%s); "
            "retrying with plain remark",
            exc,
        )

    try:
        receipt, call_function = await submit_remark(
            client, keystore.signer, payload, on_fallback=_warn_fallback,
        )
        if receipt.error:
            return False, f"{call_function} rejected: {receipt.error}"
        _AUTO_IDENTIFY_LOGGER.info(
            "auto-identify submitted: account=%s call=System.%s "
            "extrinsic=%s payload_size=%d payload_hash=0x%s",
            keystore.signer.ss58_address(),
            call_function,
            receipt.extrinsic_hash,
            len(payload),
            payload_hash,
        )
        return True, ""
    except Exception as exc:  # noqa: BLE001 — observability path; caller retries
        return False, f"{type(exc).__name__}: {exc}"


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


def _load_or_fail(
    loader: Callable[..., Any],
    config_path: Optional[str],
    raw: Optional[dict],
    empty: Any,
) -> Any:
    """Invoke ``loader`` with either ``raw=`` or a resolved path; wrap errors.

    If *raw* is provided the dict is passed directly (avoids a re-read).
    If *config_path* is ``None`` the caller-supplied *empty* default is
    returned immediately. Both live call-sites catch :class:`MinerConfigError`
    and re-raise as :class:`click.ClickException` so the CLI formats the
    message as ``quip-miner: error: …``.
    """
    if raw is not None:
        try:
            return loader(raw=raw)
        except MinerConfigError as exc:
            raise click.ClickException(str(exc)) from exc
    if config_path is None:
        return empty
    try:
        return loader(Path(config_path).expanduser())
    except MinerConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_runtime_config(
    *,
    config_path: Optional[str],
    cli_kwargs: dict,
    defaults: Optional[dict] = None,
    raw: Optional[dict] = None,
) -> dict:
    """Merge TOML config (if any) with CLI kwargs and run validate_merged.

    Precedence: CLI kwargs > TOML > `defaults`. Defaults apply *after*
    merge so a default-valued CLI arg never shadows a TOML override
    (e.g. ~/.quip-miner/signing.json as the CLI fallback must not win
    against a TOML `signer_key = "/tmp/k.json"`).

    Pass `raw` (a pre-parsed config from :func:`load_toml`) to reuse a single
    parse; otherwise the file at `config_path` is read.

    Returns the merged dict on success. Raises `click.ClickException` (so
    `miner_main`'s wrapper formats it as `quip-miner: error: ...`) when
    the TOML can't be loaded or required keys (`validators`, `signer_key`)
    are missing.
    """
    toml_data = _load_or_fail(load_miner_config, config_path, raw, {})
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
        raise _validators_unreachable(exc) from exc
    return client


async def _ensure_funded_or_fail(
    client: SubstrateClient,
    keystore,
    *,
    faucet_url: Optional[str],
    min_balance: int,
) -> int:
    """Guard C — wallet funded.

    Delegates to `shared.miner_bootstrap.ensure_funded_via_faucet`: returns the
    balance when it is already at/above threshold or after a successful faucet
    top-up. `Underfunded` (below threshold, no faucet) → `wallet-underfunded`;
    a dropped validator rotation mid-settlement → `validators-unreachable`;
    any other faucet failure → `wallet-faucet-failed` with the detail attached.
    """
    ss58 = keystore.signer.ss58_address()
    try:
        return await ensure_funded_via_faucet(
            client, keystore, faucet_url=faucet_url, min_balance=min_balance,
        )
    except Underfunded as exc:
        raise click.ClickException(
            f"wallet-underfunded ss58={ss58} "
            f"balance={exc.balance} threshold={exc.threshold}"
        ) from exc
    except NoValidatorReachable as exc:
        # Faucet path stays connected via SubstrateClient; if the rotation
        # collapses mid-settlement, surface it as the same code the connect
        # guard uses rather than burying it in `wallet-faucet-failed`.
        raise _validators_unreachable(exc) from exc
    except Exception as exc:  # noqa: BLE001 — translated to a CLI error code
        # Surface the exception text, not just the class name: the actionable
        # detail (e.g. "faucet returned 502: transfer failed; see faucet logs"
        # when the funder wallet is drained) lives in the message. Bare
        # "error=RuntimeError" tells an operator nothing.
        detail = str(exc).strip() or type(exc).__name__
        raise click.ClickException(
            f"wallet-faucet-failed ss58={ss58} threshold={min_balance} "
            f"error={type(exc).__name__}: {detail}"
        ) from exc


async def _ensure_registered_or_fail(client: SubstrateClient, keystore) -> None:
    """Guard D — miner registered, transparently and idempotently.

    Reuses ``shared.miner_bootstrap._ensure_registered`` so the manual
    ``bootstrap`` command and the miner's own startup register through
    identical code — they cannot drift. Retries registration over several
    minutes (verifying the account lands in ``QuantumPow.Miners``); an ultimate
    failure is fatal via the ``miner-registration-failed`` CLI error code.
    Already-registered is a no-op that just reports state.
    """
    ss58 = keystore.signer.ss58_address()

    async def _attempt():
        try:
            newly = await _ensure_registered(client, keystore)
        except NoValidatorReachable:
            raise  # distinct fatal code; not a registration-logic failure
        except Exception as exc:  # noqa: BLE001 — retryable, surfaced in detail
            detail = str(exc).strip() or type(exc).__name__
            return False, f"{type(exc).__name__}: {detail}"
        return True, ("registered" if newly else "already-registered")

    try:
        outcome = await _retry_until_verified("register-miner", _attempt)
    except NoValidatorReachable as exc:
        raise _validators_unreachable(exc) from exc
    except RuntimeError as exc:
        raise click.ClickException(
            f"miner-registration-failed ss58={ss58} error={exc}"
        ) from exc
    click.echo(
        f"registered miner: {ss58}" if outcome == "registered"
        else f"miner already registered: {ss58}"
    )


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
    setup_logging(log_level=log_level.upper())


@quip_miner.command("selftest")
def quip_miner_selftest() -> None:
    """Verify the packaged binary can load its bundled runtime assets.

    Guards against the PyInstaller regression where scalecodec's SCALE
    type-registry presets (shipped as package *data*, invisible to import
    analysis) get dropped from the frozen bundle. When that happens,
    ``SubstrateInterface.__init__`` loads preset ``core``,
    ``load_type_registry_preset`` returns ``None`` on the missing file, and
    ``update_type_registry(None)`` dies with "'NoneType' object has no
    attribute 'get'" on the first validator connect — surfacing as a
    misleading "could not reach" error.

    This replicates that exact load path with no live node, so CI catches a
    missing-data regression at build time (run on the frozen binary). Exits
    non-zero on failure.
    """
    from scalecodec.base import RuntimeConfigurationObject
    from scalecodec.type_registry import load_type_registry_preset

    runtime_config = RuntimeConfigurationObject()
    for preset_name in ("core", "legacy"):
        preset = load_type_registry_preset(preset_name)
        if preset is None:
            raise click.ClickException(
                f"scalecodec type-registry preset '{preset_name}' is missing "
                "from the bundle — collect_data_files regression in "
                "pyinstaller/quip_miner.spec"
            )
        runtime_config.update_type_registry(preset)
    click.echo("selftest OK: scalecodec type-registry presets load")


@quip_miner.command("resolve-mode")
@_config_option
@click.option(
    "--default",
    "default_mode",
    type=click.Choice(list(MODE_NAMES), case_sensitive=False),
    default=None,
    help="Mode to use when --config has no backend sections. Without "
    "this and an empty config, resolution fails with `no-mode-resolvable`.",
)
@click.option(
    "--image-supports",
    "image_supports_csv",
    default=None,
    help="Comma-separated subset of {cpu,gpu,qpu} this container is "
    "built to run. Resolution fails with `unsupported-mode` when the "
    "config asks for a mode outside this set. Set by Docker images "
    "via QUIP_IMAGE_SUPPORTS; omit for unrestricted resolution.",
)
@click.option(
    "--mine-mode",
    "mine_mode",
    type=click.Choice(["pow", "mempool", "both"], case_sensitive=False),
    default=None,
    help="See `quip-miner resolve-modes --help`. Same guard, single-mode "
    "convenience caller.",
)
def quip_miner_resolve_mode(
    config_path: Optional[str],
    default_mode: Optional[str],
    image_supports_csv: Optional[str],
    mine_mode: Optional[str],
) -> None:
    """Resolve the quip-miner subcommand for a config file.

    Reads the v0.1-shape backend tables (`[cpu]`, `[gpu]`, `[cuda.N]`,
    `[metal]`, `[modal]`, `[qpu]`, `[dwave]`, `[ibm]`, `[braket]`,
    `[pasqal]`, `[ionq]`, `[origin]`) and prints the matching subcommand
    on stdout (one of cpu/gpu/qpu). Exits non-zero with a kebab-case
    error code on stderr when resolution is ambiguous, unsupported by
    the image, or unspecified.

    Designed for entrypoint scripts that need to dispatch:

        MODE=$(quip-miner resolve-mode --config /data/config.toml \
                  --default cpu --image-supports cpu,qpu) || exit 1
        exec quip-miner "$MODE" --config /data/config.toml ...
    """
    backends = _load_backends_or_fail(config_path)
    image_supports = _parse_image_supports(image_supports_csv)

    try:
        resolved = resolve_mode(
            backends,
            default=default_mode.lower() if default_mode else None,
            image_supports=image_supports,
            mine_mode=mine_mode.lower() if mine_mode else None,
        )
    except ModeResolutionError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(resolved)


@quip_miner.command("resolve-modes")
@_config_option
@click.option(
    "--default",
    "default_mode",
    type=click.Choice(list(MODE_NAMES), case_sensitive=False),
    default=None,
    help="Mode to use when --config has no backend sections.",
)
@click.option(
    "--image-supports",
    "image_supports_csv",
    default=None,
    help="Comma-separated subset of {cpu,gpu,qpu} this container can run.",
)
@click.option(
    "--mine-mode",
    "mine_mode",
    type=click.Choice(["pow", "mempool", "both"], case_sensitive=False),
    default=None,
    help="Work source the miner will run with (mirrors quip-miner cpu/gpu/qpu's "
    "--mode flag). When mempool or both, a multi-backend config is rejected with "
    "`multi-backend-not-allowed-in-mempool-mode` — a single substrate account can "
    "only register as one solver type, so the other children would silently fail "
    "registration.",
)
def quip_miner_resolve_modes(
    config_path: Optional[str],
    default_mode: Optional[str],
    image_supports_csv: Optional[str],
    mine_mode: Optional[str],
) -> None:
    """Resolve the quip-miner subcommand(s) for a config file.

    Like `resolve-mode` but returns the *full* list — one mode per
    line — when the config declares multiple backend groups. The
    Docker entrypoint uses this to drive the per-mode child-process
    supervisor:

        mapfile -t MODES < <(quip-miner resolve-modes --config $CFG ...)
        for mode in "${MODES[@]}"; do
            quip-miner "$mode" --config "$CFG" ... &
        done
    """
    backends = _load_backends_or_fail(config_path)
    image_supports = _parse_image_supports(image_supports_csv)

    try:
        modes = resolve_modes(
            backends,
            default=default_mode.lower() if default_mode else None,
            image_supports=image_supports,
            mine_mode=mine_mode.lower() if mine_mode else None,
        )
    except ModeResolutionError as exc:
        raise click.ClickException(str(exc)) from exc

    for mode in modes:
        click.echo(mode)


@quip_miner.command("telemetry")
@click.option(
    "--snapshot-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing per-mode telemetry-stats-*.json files. "
    "The aggregator globs every matching file and merges them via "
    "shared.stats_snapshot.merge_snapshots on each /api/v1/* request.",
)
@click.option(
    "--rest-host",
    default="0.0.0.0",
    show_default=True,
    help="aiohttp bind address.",
)
@click.option(
    "--rest-port",
    type=int,
    default=8086,
    show_default=True,
    help="aiohttp listen port.",
)
@click.option(
    "--validator",
    "validators",
    multiple=True,
    help="Optional validator WebSocket URL for chain-backed endpoints "
    "(/api/v1/block/*, /api/v1/solve). Repeatable for failover. Omit "
    "if the aggregator should only surface controller state, not "
    "chain reads.",
)
def quip_miner_telemetry(
    snapshot_dir: str,
    rest_host: str,
    rest_port: int,
    validators: tuple,
) -> None:
    """Run the standalone telemetry aggregator.

    Used by the Docker entrypoint to host a single /api/v1 surface
    for a multi-process container. Each `quip-miner <mode>` child
    writes its own snapshot file into `snapshot_dir`; this process
    reads all of them on every request and merges into a unified
    view (summed counters, unioned miners, first-non-empty
    descriptor/survey/identity).

    Standalone — no controller, no MinerCore, no signer. Pure
    aggregator + chain-read proxy.
    """
    import multiprocessing as mp
    from shared.telemetry_process import telemetry_main

    snap_dir = Path(snapshot_dir).expanduser()
    snap_dir.mkdir(parents=True, exist_ok=True)

    shutdown_event = mp.Event()
    # Hook SIGINT here so a Ctrl-C in standalone runs exits cleanly;
    # under the Docker supervisor, SIGTERM from the entrypoint's
    # `kill -TERM` does the same via telemetry_main's own handler.
    def _on_sigint(*_a):
        shutdown_event.set()
    signal.signal(signal.SIGINT, _on_sigint)

    telemetry_main(
        listen_host=rest_host,
        listen_port=int(rest_port),
        snapshot_dir=str(snap_dir),
        validator_urls=list(validators),
        shutdown_event=shutdown_event,
    )


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
        raise _validators_unreachable(exc) from exc

    click.echo("bootstrap complete")
    click.echo(f"  ss58 address       : {result.ss58_address}")
    click.echo(f"  account_id         : {result.account_id_hex}")
    click.echo(f"  balance (plancks)  : {result.balance_plancks}")
    click.echo(f"  miner registered   : {result.miner_registered}")
    click.echo(f"  topology seeded    : {result.topology_seeded}")
    click.echo(f"  difficulty seeded  : {result.difficulty_seeded}")


# The faucet bot lives in its own repository (`gitlab.com/quip.network/faucet`)
# so it can be deployed and versioned independently of `quip-protocol`. The
# miner only speaks to it over HTTP (`--faucet-url`); see that repo for the
# server's CLI surface and the `/request` contract `_post_faucet` targets.


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

            receipt = await client.register_solver(keystore.signer, mt)
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

            receipt = await client.deregister_solver(keystore.signer)
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


# Seconds to wait for a controller to drain after signalling shutdown.
# Must exceed the longest controller poll interval (<=10s for head-subscription
# timeout in SubstrateMinerController).
_SHUTDOWN_GRACE_SECONDS = 15.0


def _parse_config_or_fail(config_path: Optional[str]) -> dict:
    """Parse the --config TOML once (``{}`` when no --config).

    Lets a command parse the file a single time and feed the result to the
    several ``load_*`` views (miner table, backends, submission) via their
    ``raw=`` parameter instead of re-reading + re-parsing it three times.
    Loader errors surface as the CLI's standard `quip-miner: error: ...`.
    """
    if config_path is None:
        return {}
    try:
        return dict(load_toml(Path(config_path).expanduser()))
    except MinerConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _load_backends_or_fail(
    config_path: Optional[str], *, raw: Optional[dict] = None
) -> dict:
    """Read the v0.1-shape backend tables from --config; `{}` if no --config.

    Wraps `load_backend_config` so loader errors surface as the same
    `quip-miner: error: <code>` formatting the rest of the CLI uses
    (via `miner_main`'s wrapper). Pass `raw` (a pre-parsed config from
    :func:`load_toml`) to reuse a single parse.
    """
    return _load_or_fail(load_backend_config, config_path, raw, {})


def _parse_image_supports(image_supports_csv: Optional[str]) -> Optional[list]:
    """Parse a `--image-supports cpu,gpu,qpu` CSV into a lowercased list."""
    if image_supports_csv is None:
        return None
    return [
        tok.strip().lower()
        for tok in image_supports_csv.split(",")
        if tok.strip()
    ]


def _load_submission_config_or_default(
    config_path: Optional[str], *, raw: Optional[dict] = None
) -> SubmissionConfig:
    """Read the ``[submission]`` table from --config; defaults if no --config.

    Wraps `load_submission_config` so a malformed knob surfaces as the
    CLI's standard error formatting rather than a raw traceback. Pass `raw`
    (a pre-parsed config from :func:`load_toml`) to reuse a single parse.
    """
    return _load_or_fail(load_submission_config, config_path, raw, SubmissionConfig())


def _check_backend_conflicts(
    ctx: click.Context,
    backends: dict,
    *,
    group: str,
    flag_param_pairs: tuple,
) -> None:
    """Fail fast when TOML defines a backend section and a conflicting
    CLI flag was passed.

    Rule (per operator request): TOML and CLI must not both describe
    the same backend — there's no ambiguity-resolving precedence, the
    operator picks one source. Click's `get_parameter_source` is what
    tells us a value came from `COMMANDLINE` vs a `DEFAULT` filled in
    by Click itself; we only error on the former so default-valued
    flags don't poison a TOML-driven run.

    `flag_param_pairs` is `((cli_label, click_param_name), ...)` — the
    label is what we surface to the operator (e.g. `--num-cpus`), the
    param name is the function-arg key Click registers.
    """
    sections = present_backend_groups(backends).get(group, [])
    if not sections:
        return

    explicit_flags: list[str] = []
    for cli_label, param_name in flag_param_pairs:
        source = ctx.get_parameter_source(param_name)
        if source == click.core.ParameterSource.COMMANDLINE:
            explicit_flags.append(cli_label)
    if not explicit_flags:
        return

    section_csv = ", ".join(f"[{s}]" for s in sections)
    flag_csv = " ".join(explicit_flags)
    raise click.UsageError(
        f"config-conflict: --config defines {section_csv} but also passed "
        f"{flag_csv}. Pick one source for {group.upper()} inventory — remove "
        f"the conflicting CLI flag(s) or drop the TOML section."
    )


def _telemetry_external_via_env() -> bool:
    """True when an external aggregator owns the REST surface.

    Set by the Docker entrypoint when it spawns `quip-miner telemetry`
    as a separate sibling. Each `quip-miner <mode>` child then writes
    its snapshot but skips its own in-process telemetry sibling —
    otherwise N children would all try to bind the same port.
    """
    val = os.environ.get("QUIP_TELEMETRY_EXTERNAL", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _qpu_miner_kind_from_backends(backends: dict, fallback: str) -> str:
    """Derive the substrate `miner_kind` from TOML QPU sections.

    Matches the v0.2 CLI's mapping: bare `qpu` for D-Wave, `qpu_<type>`
    for IBM / IonQ / Pasqal. When TOML has multiple QPU vendors (rare
    in practice — operators typically pick one), the first match in
    section-declaration order wins. Falls back to `fallback` (the CLI
    --qpu-type value) when no recognised vendor section is present.
    """
    for section in QPU_BACKEND_SECTIONS:
        if section == "qpu":
            # `[qpu]` alone doesn't bind a vendor — keep walking for an
            # explicit `[dwave]`/`[ibm]`/etc. before falling back.
            continue
        if section in backends:
            return qpu_miner_kind(section)
    return qpu_miner_kind(fallback)


def _echo_runtime_config(
    *,
    validators: tuple,
    faucet_url: Optional[str],
    rest_host: str,
    rest_port: int,
    node_name: Optional[str],
    node_log: Optional[str],
    miner_config: dict,
    submission_config: SubmissionConfig,
) -> None:
    """Echo non-secret runtime config at startup for diagnosability.

    ``miner_config`` is passed through the shared ``_scrub`` redactor so any
    token/password/key it carries is dropped — the banner is safe to paste
    into a bug report. The ss58 address, mode, and topology are already
    printed by their own banner lines, so they are not repeated here.
    """
    click.echo("config:")
    click.echo(f"  validators={','.join(validators)}")
    click.echo(f"  faucet_url={faucet_url or '(none)'}")
    click.echo(f"  rest={rest_host}:{rest_port}")
    click.echo(f"  node_name={node_name or _default_node_name()}")
    click.echo(f"  node_log={node_log or '(stderr only)'}")
    click.echo(f"  submission={submission_config}")
    click.echo(f"  miners={_scrub(miner_config)}")


async def _run_startup_guards(
    client: SubstrateClient,
    keystore,
    *,
    faucet_url: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    miner_config: dict,
) -> None:
    """Run the fund / register / descriptor guards (C, D, E) in order.

    Each guard raises ``click.ClickException`` (or a descriptor-* code) on
    ultimate failure, so all three must pass before any controller starts.
    Extracted from :func:`_run_concurrent_miner` so the orchestration body
    stays readable.
    """
    # Guard C — wallet funded (with optional faucet top-up).
    await _ensure_funded_or_fail(
        client,
        keystore,
        faucet_url=faucet_url,
        min_balance=DEFAULT_MIN_BALANCE_PLANCKS,
    )

    # Guard D — miner registered. Self-registers on first run (after
    # funding, so the MinerDeposit reserve is covered) and is a no-op on
    # every later run. Replaces the old manual `quip-miner bootstrap`
    # prerequisite; the controller still verifies before mining.
    await _ensure_registered_or_fail(client, keystore)

    # Guard E — file the signed NodeDescriptor remark so dashboards can
    # map our AccountId → node_name + advertised hardware. Retried over
    # several minutes and fatal on failure (descriptor-* codes); there is
    # no opt-out — descriptor visibility is part of the miner contract.
    await _auto_identify(
        client,
        keystore,
        node_name=node_name,
        public_host=public_host,
        public_port=public_port,
        log_level=None,
        miners_config=miner_config,
    )


async def _orchestrate_controllers(pow_controller, mempool_controller) -> int:
    """Run the PoW + mempool controllers until one exits, then drain.

    Installs SIGINT/SIGTERM handlers that signal both controllers to stop,
    waits for the first to finish (one failing brings the whole process down
    so operators can re-spawn), then gives the other up to
    ``_SHUTDOWN_GRACE_SECONDS`` to drain before cancelling. Returns the process
    exit code. Extracted from :func:`_run_concurrent_miner`.
    """
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


class _MiningStartupError(Exception):
    """A miner startup failure carrying a CLI exit code + stderr message.

    Raised by the ``_prepare_core`` / ``_build_controllers`` helpers so the
    distinct exit codes (2 = no handles, 3 = chain unqueryable / unseeded,
    4 = topology mismatch, 5 = `--mode both` unsplittable) are mapped in one
    place while ``_run_concurrent_miner``'s ``finally`` cleanup still runs.
    """

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _prepare_core(
    mode: str, miner_kind: str, miner_config: dict, topology
) -> Tuple[MinerCore, list, list]:
    """Build the ``MinerCore`` and split its handles for the requested mode.

    Returns ``(core, pow_handles, mempool_handles)``. Raises
    :class:`_MiningStartupError` (closing the freshly-built core first) when no
    handles were built (code 2) or ``--mode both`` can't be split across ≥2
    handles (code 5).
    """
    core = MinerCore(
        node_id=f"quip-miner-{mode}", miners_config=miner_config,
        topology=topology,
    )
    if not core.miner_handles:
        core.close()
        raise _MiningStartupError(
            2, f"no miner handles built for kind={miner_kind}"
        )
    pow_handles, mempool_handles = _split_handles_for_mode(
        mode, core.miner_handles
    )
    if mode == "both" and (not pow_handles or not mempool_handles):
        core.close()
        raise _MiningStartupError(
            5,
            f"--mode both requires at least 2 worker handles to split; "
            f"got {len(core.miner_handles)}. Increase --num-cpus / GPU "
            f"devices.",
        )
    return core, pow_handles, mempool_handles


async def _build_controllers(
    *,
    client: SubstrateClient,
    pool: ValidatorPool,
    core: MinerCore,
    keystore,
    topology,
    topology_spec: str,
    miner_kind: str,
    pow_handles: list,
    mempool_handles: list,
    telemetry_port: int,
    submission_config: SubmissionConfig,
) -> Tuple[Optional[SubstrateMinerController], Optional[MempoolMinerController]]:
    """Resolve the topology binding and build the PoW / mempool controllers.

    Returns ``(pow_controller, mempool_controller)``; either is ``None`` per the
    handle split. The canonical-hash recipe lives on the client
    (:meth:`SubstrateClient.resolve_topology_binding`) so this layer just reacts
    to the result. Raises :class:`_MiningStartupError` when the chain has no
    registered topology or a state query fails (code 3), or when the PoW
    sampler topology doesn't match the chain's registered one (code 4).
    """
    binding = None
    if pow_handles or mempool_handles:
        try:
            binding = await client.resolve_topology_binding(
                topology,
                miner_account_bytes=keystore.signer.account_id_bytes(),
            )
        except NoRegisteredTopology as exc:
            raise _MiningStartupError(
                3,
                "chain has no registered topology; run "
                "`quip-miner bootstrap --seed-chain` first",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surfaced as a CLI code
            raise _MiningStartupError(
                3, f"failed to query chain state: {exc}",
            ) from exc

    pow_controller = None
    mempool_controller = None

    if pow_handles:
        # PoW requires the sampler's topology to match the chain's registered
        # DefaultTopology (the chain validates this in `submit_proof` via
        # `InvalidTopology`).
        if not binding.matches:
            raise _MiningStartupError(
                4,
                f"PoW mode topology mismatch: --topology {topology_spec} "
                f"hashes to 0x{binding.expected_hash.hex()} but chain has "
                f"0x{binding.chain_hash.hex()}",
            )
        pow_controller = SubstrateMinerController(
            pool=pool,
            signer=keystore.signer,
            miner_handles=pow_handles,
            topology_hash=binding.chain_hash,
            core=core,
            telemetry_port=telemetry_port,
            # Tag the per-process snapshot file with the miner_kind so
            # multi-process containers don't race the same path; kind is
            # fixed for this controller's lifetime.
            snapshot_kind=miner_kind,
            # The Docker entrypoint sets QUIP_TELEMETRY_EXTERNAL=1 when a
            # separate `quip-miner telemetry` aggregator owns the REST
            # surface — controllers then skip the in-process sibling spawn
            # but still write snapshots.
            spawn_telemetry_sibling=not _telemetry_external_via_env(),
            submission_config=submission_config,
        )
        click.echo(
            f"  pow handles: {[h.miner_id for h in pow_handles]} "
            f"topology=0x{binding.chain_hash.hex()[:16]}..."
        )

    if mempool_handles:
        # Bind the mempool sampler to the canonical local hash the chain
        # expects (computed once in the binding above).
        snapshot = binding.snapshot
        mempool_controller = MempoolMinerController(
            pool=pool,
            signer=keystore.signer,
            miner_handles=mempool_handles,
            sampler_topology_hash=binding.expected_hash,
            allowed_h_values=snapshot.allowed_h_values,
            allowed_j_values=snapshot.allowed_j_values,
            allowed_spin_values=snapshot.allowed_spin_values,
            solver_type=MinerType.from_kind(miner_kind),
            core=core,
        )
        click.echo(
            f"  mempool handles: {[h.miner_id for h in mempool_handles]} "
            f"topology=0x{binding.expected_hash.hex()[:16]}..."
        )

    return pow_controller, mempool_controller


def _announce_and_load(
    *,
    mode: str,
    validators: tuple,
    signer_key_path: str,
    faucet_url: Optional[str],
    rest_port: int,
    rest_host: str,
    topology_spec: str,
    miner_config: dict,
    node_name: Optional[str],
    node_log: Optional[str],
    submission_config: SubmissionConfig,
) -> Tuple[object, object]:
    """Print the startup banner and load the keystore + topology.

    Attaches the optional rotating file log, loads the hybrid keystore
    (Guard A — raises ``click.ClickException`` when missing/unreadable), parses
    the topology, and echoes the resolved runtime config. Returns
    ``(keystore, topology)``.
    """
    click.echo(
        f"quip-miner version {get_version()} (protocol {PROTOCOL_VERSION})"
    )
    # node_log is opt-in: re-run setup_logging so a RotatingFileHandler is
    # attached alongside the click group's stderr handler (level preserved).
    if node_log:
        setup_logging(
            log_level=logging.getLevelName(logging.getLogger().level),
            node_log_file=node_log,
        )
    keystore = _load_keystore_or_fail(signer_key_path)
    click.echo(f"signer: {keystore.signer.ss58_address()} (hybrid) mode={mode}")
    topology = _parse_topology(topology_spec)
    click.echo(
        f"topology: {topology_spec} ({topology.num_nodes} nodes, "
        f"{topology.num_edges} edges)"
    )
    _echo_runtime_config(
        validators=validators,
        faucet_url=faucet_url,
        rest_host=rest_host,
        rest_port=rest_port,
        node_name=node_name,
        node_log=node_log,
        miner_config=miner_config,
        submission_config=submission_config,
    )
    return keystore, topology


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
    submission_config: Optional[SubmissionConfig] = None,
) -> int:
    """Unified entry for `--mode pow|mempool|both` on cpu/gpu/qpu.

    Runs the linear startup sequence — announce + load identity
    (:func:`_announce_and_load`), build workers (:func:`_prepare_core`),
    connect + guards, build controllers (:func:`_build_controllers`) — then
    hands off to :func:`_orchestrate_controllers`. Startup failures raise
    :class:`_MiningStartupError` (mapped to a CLI exit code below); the
    ``finally`` tears down whatever was constructed.
    """
    mode = mode.lower()
    if mode not in ("pow", "mempool", "both"):
        click.echo(f"invalid --mode {mode!r}", err=True)
        return 2

    # Submission tuning (tip + retry bounds). Defaults reproduce pre-tip
    # behavior, so an old config or a caller that omits this is unchanged.
    submission_config = submission_config or SubmissionConfig()
    keystore, topology = _announce_and_load(
        mode=mode,
        validators=validators,
        signer_key_path=signer_key_path,
        faucet_url=faucet_url,
        rest_port=rest_port,
        rest_host=rest_host,
        topology_spec=topology_spec,
        miner_config=miner_config,
        node_name=node_name,
        node_log=node_log,
        submission_config=submission_config,
    )

    # Declare all resources before the try so the finally block can always
    # clean up whatever was partially constructed on setup failure.
    core: Optional[MinerCore] = None
    pool: Optional[ValidatorPool] = None
    pow_controller = None
    mempool_controller = None
    setup_client: Optional[SubstrateClient] = None
    try:
        core, pow_handles, mempool_handles = _prepare_core(
            mode, miner_kind, miner_config, topology
        )

        pool = ValidatorPool(urls=tuple(validators))
        # Guard B — validators reachable. The setup client is a direct
        # SubstrateClient used for the startup sequence (Guards B/C/D/E and
        # the initial topology-snapshot read). It's closed before controllers
        # start; the controllers each own their own in-parent build_client and
        # route reads + submissions through the swap-aware pool.
        client = setup_client = await _connect_or_fail(tuple(validators))
        # Guards C/D/E — funded, registered, descriptor filed. Each raises a
        # CLI error code on ultimate failure; all three pass before any
        # controller starts.
        await _run_startup_guards(
            client,
            keystore,
            faucet_url=faucet_url,
            node_name=node_name,
            public_host=public_host,
            public_port=public_port,
            miner_config=miner_config,
        )

        # Resolve telemetry port. The sibling process is unconditional;
        # any non-positive `rest_port` (legacy `-1` default, `0`, `None`)
        # collapses to 8086.
        telemetry_port = rest_port if rest_port and rest_port > 0 else 8086

        pow_controller, mempool_controller = await _build_controllers(
            client=client,
            pool=pool,
            core=core,
            keystore=keystore,
            topology=topology,
            topology_spec=topology_spec,
            miner_kind=miner_kind,
            pow_handles=pow_handles,
            mempool_handles=mempool_handles,
            telemetry_port=telemetry_port,
            submission_config=submission_config,
        )

        click.echo(
            f"telemetry api: http://{rest_host}:{telemetry_port}/api/v1/status "
            "(sibling process)"
        )

        # Setup done — close the direct client so controllers own the
        # only live parent-side connections from here on.
        await setup_client.close()
        setup_client = None

        return await _orchestrate_controllers(
            pow_controller, mempool_controller
        )

    except _MiningStartupError as exc:
        click.echo(exc.message, err=True)
        return exc.code
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
        if core is not None:
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

# Default miner_config for each --gpu-backend choice (no TOML inventory).
_GPU_BACKEND_DEFAULTS: dict[str, dict] = {
    "local": {"cuda": [{"device": "0"}]},
    "metal": {"metal": [{}]},
    "modal": {"modal": [{"gpu_type": "t4"}]},
}

# Shared default fallbacks for the cpu/gpu/qpu commands (applied after the
# TOML+CLI merge, so an explicit TOML/CLI value always wins).
_MINING_DEFAULTS = {
    "signer_key": "~/.quip-miner/signing.json",
    "rest_port": -1,
    "rest_host": "127.0.0.1",
}


def _mining_common_options(f):
    """Options + guards shared verbatim by the cpu/gpu/qpu mining commands.

    Bundles the validator / config / faucet / signer-key / mode / topology /
    rest / identification options so each command declares only its
    backend-specific flags. Apply *above* ``@click.pass_context`` so ``ctx``
    stays the command's first positional argument.
    """
    f = _identification_options(f)
    f = click.option(
        "--rest-host", type=str, default=None, show_default=False,
        help=_REST_HOST_HELP,
    )(f)
    f = click.option(
        "--rest-port", type=int, default=None, show_default=False,
        help=_REST_PORT_HELP,
    )(f)
    f = click.option(
        "--topology", "topology_spec", default="advantage2_system1",
        show_default=True, help=_TOPOLOGY_HELP,
    )(f)
    f = click.option(
        "--mode",
        type=click.Choice(["pow", "mempool", "both"], case_sensitive=False),
        default="pow", show_default=True, help=_MODE_HELP,
    )(f)
    f = click.option(
        "--signer-key", "signer_key_path", type=click.Path(dir_okay=False),
        default=None,
        help=_SIGNER_KEY_HELP + " Falls back to --config `signer_key`, "
        "then ~/.quip-miner/signing.json.",
    )(f)
    f = _faucet_url_option(f)
    f = _config_option(f)
    f = _validator_option(f)
    return f


def _dispatch_mining_command(
    *,
    raw: dict,
    miner_kind: str,
    miner_config: dict,
    validators: tuple,
    signer_key_path: Optional[str],
    faucet_url: Optional[str],
    mode: str,
    topology_spec: str,
    rest_port: Optional[int],
    rest_host: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    node_log: Optional[str],
) -> None:
    """Shared tail of the cpu/gpu/qpu commands; raises ``SystemExit``.

    Merges CLI overrides onto the already-parsed `raw` config and hands off to
    :func:`_run_concurrent_miner`. Each command supplies only its
    backend-specific `miner_kind` + `miner_config`; reusing `raw` keeps the
    config file parsed exactly once per invocation.
    """
    merged = _resolve_runtime_config(
        config_path=None,
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
        defaults=_MINING_DEFAULTS,
        raw=raw,
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
        submission_config=_load_submission_config_or_default(None, raw=raw),
    )))


@quip_miner.command("cpu")
@_mining_common_options
@click.option(
    "--num-cpus",
    type=int,
    default=1,
    show_default=True,
    help="Number of CPU SA workers to spawn",
)
@click.pass_context
def quip_miner_cpu(
    ctx: click.Context,
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
    raw = _parse_config_or_fail(config_path)
    backends = _load_backends_or_fail(config_path, raw=raw)
    _check_backend_conflicts(
        ctx,
        backends,
        group="cpu",
        flag_param_pairs=(("--num-cpus", "num_cpus"),),
    )
    if "cpu" in backends:
        # TOML drives inventory verbatim — MinerCore._initialize_miners
        # reads `cpu.num_cpus` and `cpu.args` directly. Coerce the
        # tomllib Mapping → plain dict so downstream consumers can
        # mutate without surprises.
        miner_config = {"cpu": dict(backends["cpu"])}
    else:
        miner_config = {"cpu": {"num_cpus": num_cpus}}
    _dispatch_mining_command(
        raw=raw,
        miner_kind="cpu",
        miner_config=miner_config,
        validators=validators,
        signer_key_path=signer_key_path,
        faucet_url=faucet_url,
        mode=mode,
        topology_spec=topology_spec,
        rest_port=rest_port,
        rest_host=rest_host,
        node_name=node_name,
        public_host=public_host,
        public_port=public_port,
        node_log=node_log,
    )


@quip_miner.command("gpu")
@_mining_common_options
@click.option(
    "--gpu-backend",
    type=click.Choice(["local", "metal", "modal"], case_sensitive=False),
    default="local",
    show_default=True,
    help="GPU backend: local CUDA, Apple Metal, or Modal cloud",
)
@click.pass_context
def quip_miner_gpu(
    ctx: click.Context,
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
    raw = _parse_config_or_fail(config_path)
    backends = _load_backends_or_fail(config_path, raw=raw)
    _check_backend_conflicts(
        ctx,
        backends,
        group="gpu",
        flag_param_pairs=(("--gpu-backend", "gpu_backend"),),
    )
    has_gpu_toml = any(s in backends for s in GPU_BACKEND_SECTIONS)
    if has_gpu_toml:
        # TOML inventory drives the miner config — `_build_gpu_specs`
        # handles `[gpu]` defaults plus any of `[cuda.N]` / `[nvidia.N]`
        # / `[metal]` / `[modal]` device sections.
        miner_config = {
            s: backends[s] for s in GPU_BACKEND_SECTIONS if s in backends
        }
    else:
        backend = gpu_backend.lower()
        if backend not in _GPU_BACKEND_DEFAULTS:
            raise click.BadParameter(f"unknown --gpu-backend: {backend}")
        miner_config = _GPU_BACKEND_DEFAULTS[backend]

    _dispatch_mining_command(
        raw=raw,
        miner_kind="gpu",
        miner_config=miner_config,
        validators=validators,
        signer_key_path=signer_key_path,
        faucet_url=faucet_url,
        mode=mode,
        topology_spec=topology_spec,
        rest_port=rest_port,
        rest_host=rest_host,
        node_name=node_name,
        public_host=public_host,
        public_port=public_port,
        node_log=node_log,
    )


@quip_miner.command("qpu")
@_mining_common_options
@click.option(
    "--qpu-type",
    type=click.Choice(["dwave", "ibm", "ionq", "pasqal", "braket", "origin"]),
    default="dwave",
    show_default=True,
    help="QPU provider",
)
@click.option(
    "--daily-budget",
    type=str,
    default=None,
    help="Daily QPU access-time budget. Format: '30s', '5m', '2h', '1d' (passed to QPUTimeManager)",
)
@click.pass_context
def quip_miner_qpu(
    ctx: click.Context,
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
    raw = _parse_config_or_fail(config_path)
    backends = _load_backends_or_fail(config_path, raw=raw)
    _check_backend_conflicts(
        ctx,
        backends,
        group="qpu",
        flag_param_pairs=(
            ("--qpu-type", "qpu_type"),
            ("--daily-budget", "daily_budget"),
        ),
    )
    has_qpu_toml = any(s in backends for s in QPU_BACKEND_SECTIONS)
    if has_qpu_toml:
        miner_config = {
            s: backends[s] for s in QPU_BACKEND_SECTIONS if s in backends
        }
        miner_kind = _qpu_miner_kind_from_backends(backends, fallback=qpu_type)
    else:
        section: dict = {"type": qpu_type}
        if daily_budget is not None:
            section["daily_budget"] = daily_budget
        miner_config = {qpu_type: [section]}
        miner_kind = qpu_miner_kind(qpu_type)

    _dispatch_mining_command(
        raw=raw,
        miner_kind=miner_kind,
        miner_config=miner_config,
        validators=validators,
        signer_key_path=signer_key_path,
        faucet_url=faucet_url,
        mode=mode,
        topology_spec=topology_spec,
        rest_port=rest_port,
        rest_host=rest_host,
        node_name=node_name,
        public_host=public_host,
        public_port=public_port,
        node_log=node_log,
    )


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
    try:
        raw = dict(load_toml(path))
    except MinerConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    raw.pop("miner", None)
    return raw


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
    miner_specs = build_miner_specs(node_id, miners_config)

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
            def _warn_fallback(exc: Exception) -> None:
                # Some metadata caches answer "yes" but the active runtime
                # rejects the call at compose time — degrade to plain remark
                # rather than failing the whole identify.
                click.echo(
                    f"remark_with_event submission failed ({exc}); "
                    f"retrying with plain remark",
                    err=True,
                )

            receipt, call_function = await submit_remark(
                client, keystore.signer, payload, on_fallback=_warn_fallback,
            )
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
