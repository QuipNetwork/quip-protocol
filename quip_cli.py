"""Click-based CLI for quip-miner.

Provides the `quip-miner` console command with subcommands:
- keygen     - generate a fresh sr25519 signing key
- bootstrap  - fund + register a miner account against a substrate chain
- cpu / gpu / qpu - run a miner against a substrate chain

Runtime architecture is documented in ARCHITECTURE.md at the repo root.
"""

from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import logging
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import click

from dwave_topologies.topologies.json_loader import (
    load_topology,
    topology_from_nodes_edges,
)
from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate, load
from shared.logging_config import setup_logging
from substrate.mempool_stack import MempoolStack
from substrate.mempool_types import MinerType, qpu_miner_kind
from substrate.miner_bootstrap import (
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
from substrate.miner_registry import descriptor_call_params, descriptor_payload_hash
from substrate.solver_registration import SolverGuardOutcome, ensure_solver_registered
from shared.system_info import (
    DescriptorValidationError,
    _scrub,
    build_descriptor,
    to_canonical_json,
    validate_descriptor,
)
from shared.version import PROTOCOL_VERSION, get_version
from substrate.pool import ValidatorPool
from substrate.work_scheduler import WorkScheduler


_AUTO_IDENTIFY_LOGGER = logging.getLogger("quip_miner.auto_identify")
_STARTUP_LOGGER = logging.getLogger("quip_miner.startup")

# Funding, registration, and descriptor filing must each succeed before the
# miner runs — they retry this many times with exponential backoff (delays
# 5,10,20,40,45,45,45s ≈ 3.5 minutes) before failing the startup hard. The
# node descriptor is filed right after register_miner, so its extrinsic can
# race the registration extrinsic's nonce (stale `accountNextIndex` for
# ~a block); the retry rides that out. The per-round participation marker is
# separate and stays best-effort — it is not gated here.
_STARTUP_RETRY_ATTEMPTS = 8
_STARTUP_RETRY_BASE_DELAY_SECONDS = 5.0
_STARTUP_RETRY_MAX_DELAY_SECONDS = 45.0

# After an in-block set_descriptor receipt, poll chain state until the
# stored payload_hash matches what we submitted (an in-block receipt on a
# fork that never finalizes would otherwise silently drop the descriptor —
# see the spec-110 upgrade wipe post-mortem). ~15s covers a couple of
# blocks; if the record still isn't visible the whole submit is retried
# by `_retry_until_verified`.
_DESCRIPTOR_VERIFY_POLLS = 5
_DESCRIPTOR_VERIFY_DELAY_SECONDS = 3.0


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
    """Guard E — ensure the chain holds this node's current NodeDescriptor.

    Called on every startup after registration. The on-chain record is
    compared first (payload_hash of the would-be submission vs storage):
    when it already matches, nothing is filed; when it differs or is absent
    — fresh miner, config change, or the chain forgot it (storage wipe on a
    runtime upgrade, orphaned fork) — the descriptor is submitted and the
    attempt only counts once storage actually shows it. Filing is a fatal
    startup requirement: a descriptor that can't be built/validated fails
    immediately (operator misconfiguration), and submission is retried over
    several minutes before failing hard via the ``descriptor-failed`` CLI
    code. The per-round "participating" marker is a separate, best-effort
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

    # Post the richest descriptor the live runtime accepts: V2 (system_info +
    # runtime) where available, V1 on older runtimes. The chain upgrades
    # runtime-first, so this picks up V2 automatically once it's deployed.
    schema_version = await client.descriptor_schema_version()
    try:
        call_params = descriptor_call_params(
            descriptor, node_id=effective_name, schema_version=schema_version,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"descriptor-invalid ss58={keystore.signer.ss58_address()} "
            f"error={exc}"
        ) from exc

    account = keystore.signer.account_id_bytes()
    current, local_hash = await _descriptor_already_current(
        client, account, call_params,
    )
    if current:
        _AUTO_IDENTIFY_LOGGER.info(
            "descriptor already current on chain "
            "(account=%s payload_hash=0x%s); skipping submit",
            keystore.signer.ss58_address(),
            local_hash.hex() if local_hash else "?",
        )
        return

    # Retry submission over several minutes: an extrinsic fired right after
    # register_miner can be rejected on a stale nonce until the registration
    # block is imported. Every in-block receipt is then verified against
    # chain state — the attempt only counts as done once NodeDescriptors
    # actually shows our record. Filing the descriptor is a fatal startup
    # requirement.
    async def _attempt():
        ok, detail = await _submit_descriptor(
            client, keystore, call_params=call_params
        )
        if not ok:
            return False, detail
        return await _descriptor_visible_on_chain(client, account, local_hash)

    try:
        await _retry_until_verified("file-descriptor", _attempt)
    except RuntimeError as exc:
        raise click.ClickException(
            f"descriptor-failed ss58={keystore.signer.ss58_address()} "
            f"error={exc}"
        ) from exc


async def _descriptor_already_current(
    client: SubstrateClient,
    account: bytes,
    call_params: dict,
) -> Tuple[bool, Optional[bytes]]:
    """Compare the would-be ``set_descriptor`` submission against chain state.

    The runtime stores ``blake2_256`` of the SCALE-encoded call argument as
    the descriptor's ``payload_hash``; computing the same digest locally
    tells us whether the chain already holds exactly this record (filing
    again would be pure churn). Returns ``(already_current, local_hash)``.
    The check is an optimization only: any failure degrades to
    ``(False, None)`` — submit unconditionally — never to blocking startup.
    """
    try:
        encoded = await client.encode_call_args(
            "MinerRegistry", "set_descriptor", call_params,
        )
        local_hash = descriptor_payload_hash(encoded)
        chain_hash = await client.query_descriptor_payload_hash(account)
        return chain_hash == local_hash, local_hash
    except Exception as exc:  # noqa: BLE001 — pre-check must not block startup
        _AUTO_IDENTIFY_LOGGER.warning(
            "descriptor pre-check unavailable (%s: %s); submitting unconditionally",
            type(exc).__name__, exc,
        )
        return False, None


async def _descriptor_visible_on_chain(
    client: SubstrateClient,
    account: bytes,
    local_hash: Optional[bytes],
) -> Tuple[bool, str]:
    """Poll storage until ``NodeDescriptors[account]`` reflects our submission.

    Returns ``(True, "")`` once the stored payload_hash equals *local_hash*
    (or, when the local digest couldn't be computed, once any descriptor is
    present). Returns ``(False, detail)`` after the polls are exhausted so
    the caller re-submits — an in-block receipt whose block gets orphaned
    leaves no state behind, and only this read-back catches that.
    """
    detail = "descriptor not in chain state yet"
    for i in range(_DESCRIPTOR_VERIFY_POLLS):
        if i:
            await asyncio.sleep(_DESCRIPTOR_VERIFY_DELAY_SECONDS)
        try:
            seen = await client.query_descriptor_payload_hash(account)
        except Exception as exc:  # noqa: BLE001 — poll again, then re-submit
            detail = f"verify query failed: {type(exc).__name__}: {exc}"
            continue
        if seen is None:
            detail = "descriptor not in chain state yet"
            continue
        if local_hash is None or seen == local_hash:
            return True, ""
        detail = f"chain shows different payload_hash 0x{seen.hex()}"
    return False, detail


async def _submit_descriptor(
    client: SubstrateClient,
    keystore,
    *,
    call_params: dict,
) -> Tuple[bool, str]:
    """Submit the descriptor extrinsic once.

    Returns ``(True, "")`` when it lands in-block, else ``(False, detail)``
    so the caller can back off and retry.
    """
    try:
        receipt = await client.submit_extrinsic(
            "MinerRegistry",
            "set_descriptor",
            call_params,
            keystore.signer,
            wait_for="inblock",
        )
        if receipt.error:
            return False, f"set_descriptor rejected: {receipt.error}"
        _AUTO_IDENTIFY_LOGGER.info(
            "auto-identify submitted: account=%s call=MinerRegistry.set_descriptor "
            "extrinsic=%s",
            keystore.signer.ss58_address(),
            receipt.extrinsic_hash,
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
    "DO NOT USE for live mining. The live miner pulls its topology from the "
    "chain's registered DefaultTopology automatically (and re-binds when the "
    "chain changes it), so passing --topology to a mining command is an error. "
    "This flag exists only for standalone tools / isolated analysis that run "
    "without a chain (e.g. 'zephyr:M,T' synthetic graphs or a hardware name)."
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

    Delegates to `substrate.miner_bootstrap.ensure_funded_via_faucet`: returns the
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

    Reuses ``substrate.miner_bootstrap._ensure_registered`` so the manual
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


def _importable(name: str) -> bool:
    """True when `import name` would succeed (spec found; not imported)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _detect_image_supports() -> list[str]:
    """Probe which backend groups this installation has libraries for.

    Replaces the Dockerfile-baked QUIP_IMAGE_SUPPORTS env: the install
    itself knows what it can run. cpu is always supported; gpu needs any
    of cupy (CUDA) / Metal (Apple) / modal (remote); qpu needs the
    D-Wave Ocean SDK.
    """
    supports = ["cpu"]
    if any(_importable(m) for m in ("cupy", "Metal", "modal")):
        supports.append("gpu")
    if _importable("dwave.system"):
        supports.append("qpu")
    return supports


def _plan_processes(config_path: Path) -> Tuple[Path, list[dict]]:
    """Turn a config.toml into the process set the supervisor runs.

    Pure planning — no spawns. Returns ``(runtime_dir, procs)`` where each
    proc is ``{"args": [subcommand, ...], "env": {...}}``:

    - one ``telemetry`` aggregator when the config's ``rest_port`` > 0
      (the /api/v1 surface is config-driven; -1 disables it), and
    - one miner child per backend group the config declares (the same
      resolution `resolve-modes` exposes for testing), rejected up front
      with ``unsupported-mode`` when the install lacks the libraries.

    ``runtime_dir`` (`<config dir>/runtime`) is internal glue: children
    write per-mode snapshots there and skip their in-process REST sibling;
    the aggregator reads it. Operators never configure it.
    """
    config_path = Path(config_path).expanduser()
    raw = _load_or_fail(load_toml, str(config_path), None, {})
    miner_toml = _load_or_fail(load_miner_config, None, raw, {})
    backends = _load_backends_or_fail(None, raw=raw)
    toml_mode = miner_toml.get("mode")
    try:
        modes = resolve_modes(
            backends,
            image_supports=_detect_image_supports(),
            mine_mode=str(toml_mode).lower() if toml_mode else None,
        )
    except ModeResolutionError as exc:
        raise click.ClickException(str(exc)) from exc

    runtime_dir = config_path.resolve().parent / "runtime"
    procs: list[dict] = []
    rest_port = miner_toml.get("rest_port")
    telemetry_on = isinstance(rest_port, int) and rest_port > 0
    if telemetry_on:
        procs.append({
            "args": [
                "telemetry", "--config", str(config_path),
                "--snapshot-dir", str(runtime_dir),
            ],
            "env": {},
        })
    for mode in modes:
        procs.append({
            "args": [mode, "--config", str(config_path)],
            "env": {
                "QUIP_RUNTIME_DIR": str(runtime_dir),
                "QUIP_TELEMETRY_EXTERNAL": "1",
            } if telemetry_on else {},
        })
    return runtime_dir, procs


def _run_supervisor(config_path: Path) -> int:
    """Production entry: spawn + supervise everything the config declares.

    One child per planned process (see :func:`_plan_processes`), launched
    as `quip-miner <subcommand> --config <file>`. SIGTERM/SIGINT fan out
    to every child; the first child to exit (clean or otherwise) tears
    down the rest and its code becomes the supervisor's — operators or
    orchestrators (compose/k8s/systemd) re-spawn from there.
    """
    runtime_dir, procs = _plan_processes(config_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    binary = shutil.which("quip-miner") or sys.argv[0]

    children: list[subprocess.Popen] = []
    for spec in procs:
        click.echo(f"supervisor: starting quip-miner {' '.join(spec['args'])}")
        children.append(subprocess.Popen(
            [binary, *spec["args"]],
            env={**os.environ, **spec["env"]},
        ))

    def _forward(_signum, _frame):
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    exit_code = 0
    running = True
    while running:
        for child in children:
            code = child.poll()
            if code is not None:
                exit_code = code
                running = False
                break
        else:
            time.sleep(0.5)
    click.echo(
        f"supervisor: a child exited with code {exit_code}; "
        "tearing down siblings",
        err=True,
    )
    _forward(None, None)
    deadline = time.monotonic() + 30.0
    for child in children:
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
    return exit_code


@click.group(name="quip-miner", invoke_without_command=True)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level for quip-miner subcommands",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Production mode: run everything this config.toml declares — "
    "one miner process per backend section, plus the telemetry "
    "aggregator when rest_port > 0 — and supervise them. The "
    "subcommands below are test/ops tooling.",
)
@click.pass_context
def quip_miner(ctx: click.Context, log_level: str, config_path: Optional[str]) -> None:
    """Substrate-integrated quantum mining frontend.

    Production usage is config-driven: `quip-miner --config config.toml`
    starts the full node (miners + telemetry) from the file alone.
    """
    setup_logging(log_level=log_level.upper())
    if ctx.invoked_subcommand is not None:
        return
    if config_path is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
    raise SystemExit(_run_supervisor(Path(config_path).expanduser()))


def _selftest_spawn_child(result_queue) -> None:
    """Spawn-child target for the selftest round-trip (must be module-level
    so the spawn child can unpickle it by qualified name)."""
    result_queue.put("spawn-ok")


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

    # Spawn round-trip: re-executes the frozen bootloader and all PyInstaller
    # runtime hooks in a child process — the exact path miner workers take via
    # shared/proc_util.spawn_worker. Catches hook/bundle regressions that only
    # kill spawn children (e.g. the pyi_rth_pkgres hook failing on a missing
    # setuptools vendor data file, surfacing as BrokenPipeError in production).
    import multiprocessing
    import queue as queue_mod

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    child = ctx.Process(target=_selftest_spawn_child, args=(result_queue,))
    child.start()
    try:
        token = result_queue.get(timeout=120)
    except queue_mod.Empty:
        token = None
    child.join(timeout=10)
    if token != "spawn-ok" or child.exitcode != 0:
        raise click.ClickException(
            f"multiprocessing spawn round-trip failed (exitcode={child.exitcode}, "
            f"token={token!r}) — a PyInstaller runtime hook or bundled asset is "
            "broken in spawn children; check pyinstaller/quip_miner.spec excludes"
        )
    click.echo("selftest OK: multiprocessing spawn round-trip")


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
            mine_mode=_effective_mine_mode(mine_mode, config_path),
        )
    except ModeResolutionError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(resolved)


def _effective_mine_mode(
    mine_mode: Optional[str], config_path: Optional[str]
) -> Optional[str]:
    """--mine-mode flag, falling back to the config's `[miner] mode` key.

    Keeps the mempool multi-backend guard active for config-only launches
    (the docker entrypoint passes no flags — config.toml is the single
    source of truth for the work source).
    """
    if mine_mode:
        return mine_mode.lower()
    toml_mode = _load_or_fail(load_miner_config, config_path, None, {}).get("mode")
    return str(toml_mode).lower() if toml_mode else None


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
    help="Work source override (falls back to the config's `[miner] mode` "
    "key). When mempool or both, a multi-backend config is rejected with "
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
            mine_mode=_effective_mine_mode(mine_mode, config_path),
        )
    except ModeResolutionError as exc:
        raise click.ClickException(str(exc)) from exc

    for mode in modes:
        click.echo(mode)


@quip_miner.command("telemetry")
@_config_option
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
    default=None,
    help="aiohttp bind address. Falls back to --config `rest_host`, "
    "then 0.0.0.0.",
)
@click.option(
    "--rest-port",
    type=int,
    default=None,
    help="aiohttp listen port. Falls back to --config `rest_port`, then "
    "8086. Non-positive values (the children-disabled sentinel -1) also "
    "collapse to 8086.",
)
@click.option(
    "--validator",
    "validators",
    multiple=True,
    help="Optional validator WebSocket URL for chain-backed endpoints "
    "(/api/v1/block/*, /api/v1/solve). Repeatable for failover. Falls "
    "back to --config `validators`; omit both if the aggregator should "
    "only surface controller state, not chain reads.",
)
def quip_miner_telemetry(
    config_path: Optional[str],
    snapshot_dir: str,
    rest_host: Optional[str],
    rest_port: Optional[int],
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
    aggregator + chain-read proxy. Reads rest_host/rest_port/validators
    from the same --config TOML the miner children use; flags override.
    """
    import multiprocessing as mp
    from substrate.telemetry_process import telemetry_main

    miner_toml = _load_or_fail(load_miner_config, config_path, None, {})
    effective_host = rest_host or miner_toml.get("rest_host") or "0.0.0.0"
    effective_port = rest_port if rest_port is not None \
        else miner_toml.get("rest_port")
    # The miner children treat rest_port <= 0 as "REST disabled"; the
    # aggregator exists to serve REST, so the sentinel means "default".
    if not isinstance(effective_port, int) or effective_port <= 0:
        effective_port = 8086
    effective_validators = (
        list(validators) if validators
        else [str(u) for u in miner_toml.get("validators") or []]
    )

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
        listen_host=effective_host,
        listen_port=int(effective_port),
        snapshot_dir=str(snap_dir),
        validator_urls=effective_validators,
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
        # is enough. Guard D+ does the query-first/race-tolerant work;
        # this command only maps its outcome to exit codes (details such
        # as the registered type and extrinsic hash land in the log).
        client = await _connect_or_fail(tuple(merged["validators"]))
        try:
            outcome = await ensure_solver_registered(client, keystore.signer, miner_type)
        finally:
            await client.close()
        if outcome is SolverGuardOutcome.TYPE_MISMATCH:
            click.echo(
                "solver already registered with a different type; "
                f"deregister first to change to {mt.name}",
                err=True,
            )
            return 4
        if outcome is SolverGuardOutcome.FAILED:
            click.echo("register_solver failed (see log for details)", err=True)
            return 3
        verb = (
            "already registered"
            if outcome is SolverGuardOutcome.ALREADY_REGISTERED
            else "registered"
        )
        click.echo(f"{verb} as {mt.name}")
        return 0

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
# The live miner binds its topology to the chain's registered DefaultTopology
# — it reads the snapshot's nodes/edges after connecting and builds the sampler
# from them (see _topology_from_chain), so the miner's graph can never skew
# from the chain's. When the chain changes its DefaultTopology, the controller
# signals a rebind and the supervisor re-fetches and rebuilds (no restart).
# `--topology` is therefore NOT valid for mining commands (it raises); it stays
# only for standalone tools / isolated analysis that run without a chain
# (handled by `_parse_topology`, kept for those callers).
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


def _solver_name_from_config(miner_config: dict) -> Optional[str]:
    """Return the first configured D-Wave solver name, or ``None``.

    The chain stores the topology graph but not the D-Wave solver to connect to
    (a hardware/account detail), so the QPU solver name comes from the miner
    config's ``qpu`` device block. CPU/GPU backends have no solver; ``None`` is
    fine there (the topology's ``solver_name`` is informational for them).
    """
    qpu = miner_config.get("qpu") if isinstance(miner_config, dict) else None
    if not isinstance(qpu, dict):
        return None
    for dev in qpu.get("devices", []) or []:
        if isinstance(dev, dict) and dev.get("solver"):
            return str(dev["solver"])
    if qpu.get("solver"):
        return str(qpu["solver"])
    return None


async def _topology_from_chain(client: SubstrateClient, *, account_bytes: bytes,
                               miner_config: dict):
    """Build the sampler topology from the chain's registered DefaultTopology.

    Reads the mining snapshot (nodes/edges) and constructs a topology object
    from it, so the miner's graph is the chain's by construction — no local
    file, no hash skew. Raises :class:`_MiningStartupError` (code 3) when the
    chain has no registered topology yet. ``solver_name`` comes from config.
    """
    snapshot = await client.get_mining_snapshot(miner_account_bytes=account_bytes)
    if snapshot is None:
        raise _MiningStartupError(
            3,
            "chain has no registered topology; run "
            "`quip-miner bootstrap --seed-chain` first",
        )
    solver_name = (
        _solver_name_from_config(miner_config)
        or f"chain:0x{snapshot.topology_hash.hex()[:12]}"
    )
    topology = topology_from_nodes_edges(
        snapshot.nodes, snapshot.edges, solver_name,
    )
    return topology, snapshot


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
    into a bug report. The ss58 address, mempool flag, and topology are
    already printed by their own banner lines, so they are not repeated here.
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
    mempool_enabled: bool = False,
    miner_kind: str = "cpu",
) -> bool:
    """Run the fund / register / solver / descriptor guards (C, D, D+, E).

    Guards C, D, and E raise ``click.ClickException`` (or a descriptor-*
    code) on ultimate failure, so they must pass before any mining starts.
    Guard D+ (mempool solver registration) is NON-fatal by contract: any
    outcome other than success logs loudly and disables mempool for this
    run — pow always proceeds (a fatal child exit here would trip the
    supervisor's terminate-all-siblings rule and take pow down node-wide).

    Returns the effective mempool flag: ``mempool_enabled`` AND Guard D+
    succeeded.
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

    # Guard D+ — mempool solver registered (query-first, race-tolerant,
    # never auto-deregisters). Runs only when mempool is enabled; funding
    # from Guard C covers the fee-scale cost.
    if mempool_enabled:
        mempool_enabled = await _ensure_solver_or_disable_mempool(
            client, keystore, miner_kind,
        )

    # Guard E — file the signed on-chain NodeDescriptor so dashboards can
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
    return mempool_enabled


async def _ensure_solver_or_disable_mempool(
    client: SubstrateClient, keystore, miner_kind: str
) -> bool:
    """Guard D+ — non-fatal solver registration for mempool participation.

    Returns True when the account is (now) registered with the matching
    vendor-resolved type. FAILED / TYPE_MISMATCH — and any unexpected
    exception — log loudly and return False (mempool disabled for this
    run; pow proceeds). Never raises: unlike Guard D, a failure here must
    not stop pow mining.
    """
    try:
        outcome = await ensure_solver_registered(
            client, keystore.signer, miner_kind,
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal by contract
        click.echo(
            f"mempool DISABLED for this run: solver registration guard "
            f"raised {type(exc).__name__}: {exc} — pow mining continues",
            err=True,
        )
        return False
    if outcome in (
        SolverGuardOutcome.REGISTERED,
        SolverGuardOutcome.ALREADY_REGISTERED,
    ):
        click.echo(f"mempool solver guard: {outcome.value} ({miner_kind})")
        return True
    click.echo(
        f"mempool DISABLED for this run: solver registration guard "
        f"returned {outcome.name} (kind={miner_kind}) — pow mining "
        "continues. TYPE_MISMATCH needs an explicit "
        "`quip-miner deregister-solver` + re-register; FAILED is usually "
        "an RPC/chain error (see log).",
        err=True,
    )
    return False


async def _orchestrate_controllers(
    pow_controller, scheduler, mempool_stack=None
) -> Tuple[bool, int]:
    """Run the scheduler stack until the pow controller exits, then drain.

    Starts the WorkScheduler's drainer/pump tasks, installs SIGINT/SIGTERM
    handlers that signal every component to stop, and waits FIRST_COMPLETED
    over the pow-controller + mempool-stack tasks. The pow controller
    failing brings the process down (operators re-spawn); the mempool stack
    by contract only returns on shutdown (its failures park mempool and pow
    continues). Pending tasks get up to ``_SHUTDOWN_GRACE_SECONDS`` to
    drain before cancellation. Returns ``(rebind_requested, exit_code)`` —
    ``rebind_requested`` is True when the pow controller asked to rebind to
    a changed chain topology (the caller rebuilds the stack rather than
    exiting). Extracted from :func:`_run_concurrent_miner`.
    """
    loop = asyncio.get_running_loop()

    def _signal_shutdown() -> None:
        for c in (pow_controller, mempool_stack, scheduler):
            if c is not None:
                try:
                    c.shutdown()
                except Exception:  # noqa: BLE001 -- signal handler must not raise
                    pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_shutdown)

    # Drainers must be live before the pow controller's event manager can
    # trigger the first dispatch, or an early result would be missed.
    scheduler.start()

    tasks: list[asyncio.Task] = [
        asyncio.create_task(pow_controller.run(), name="pow-controller"),
    ]
    if mempool_stack is not None:
        tasks.append(asyncio.create_task(
            mempool_stack.run(), name="mempool-stack"
        ))

    exit_code = 0
    try:
        # Wait until any task exits. The pow controller returning (clean or
        # fatal) is the process-level signal; the mempool stack only
        # returns on shutdown or a startup bug (surfaced loudly here).
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
    finally:
        await scheduler.stop()

    return (bool(getattr(pow_controller, "rebind_requested", False)), exit_code)


class _MiningStartupError(Exception):
    """A miner startup failure carrying a CLI exit code + stderr message.

    Raised by the ``_prepare_core`` / ``_build_scheduler_stack`` helpers so
    the distinct exit codes (2 = no handles, 3 = chain unqueryable /
    unseeded, 4 = topology mismatch) are mapped in one place while
    ``_run_concurrent_miner``'s ``finally`` cleanup still runs.
    """

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _prepare_core(miner_kind: str, miner_config: dict, topology) -> MinerCore:
    """Build the ``MinerCore`` whose handles the WorkScheduler will own.

    No handle split: every handle mines pow as idle filler and serves
    mempool jobs on priority preemption, so 1-handle nodes are fully
    supported. Raises :class:`_MiningStartupError` (closing the
    freshly-built core first) when no handles were built (code 2).
    """
    core = MinerCore(
        node_id="quip-miner", miners_config=miner_config, topology=topology,
    )
    if not core.miner_handles:
        core.close()
        raise _MiningStartupError(
            2, f"no miner handles built for kind={miner_kind}"
        )
    return core


async def _build_scheduler_stack(
    *,
    client: SubstrateClient,
    pool: ValidatorPool,
    core: MinerCore,
    keystore,
    topology,
    miner_kind: str,
    telemetry_port: int,
    submission_config: SubmissionConfig,
    mempool_enabled: bool,
    mempool_min_reward: int = 0,
) -> Tuple[SubstrateMinerController, WorkScheduler, Optional[MempoolStack]]:
    """Resolve the topology binding and build the single scheduler stack.

    ONE WorkScheduler owns every handle; the pow controller keeps the pow
    brain and delegates all handle operations to it; the mempool stack
    (producer + submitter glue, present only when *mempool_enabled*) feeds
    the scheduler priority jobs and consumes winning results. Both result
    consumers are queue-put callbacks per the WorkScheduler contract —
    submits run on the controller/stack loops, never on a drainer. The
    stack shares the pow controller's ONE ChainEventManager via
    ``head_subscribers``.

    Raises :class:`_MiningStartupError` when the chain has no registered
    topology or a state query fails (code 3), or when the sampler topology
    doesn't match the chain's registered one (code 4).
    """
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

    # PoW requires the sampler's topology to match the chain's registered
    # DefaultTopology (the chain validates this in `submit_proof` via
    # `InvalidTopology`).
    if not binding.matches:
        # Defensive: the topology is built from this same chain snapshot, so
        # this should never fire. If it does, the chain changed its topology
        # between the snapshot read and this binding — surface it as a
        # startup error; the next launch (or rebind) picks up the new one.
        raise _MiningStartupError(
            4,
            "topology binding race: sampler hashes to "
            f"0x{binding.expected_hash.hex()} but chain now has "
            f"0x{binding.chain_hash.hex()} (chain topology changed during "
            "startup; retry)",
        )

    mempool_stack: Optional[MempoolStack] = None
    if mempool_enabled:
        snapshot = binding.snapshot
        mempool_stack = MempoolStack(
            pool=pool,
            signer=keystore.signer,
            sampler_topology_hash=binding.expected_hash,
            allowed_h_values=snapshot.allowed_h_values,
            allowed_j_values=snapshot.allowed_j_values,
            allowed_spin_values=snapshot.allowed_spin_values,
            solver_type=MinerType.from_kind(miner_kind),
            min_reward=mempool_min_reward,
            core=core,
        )

    pow_controller = SubstrateMinerController(
        pool=pool,
        signer=keystore.signer,
        miner_handles=core.miner_handles,
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
        # ONE ChainEventManager: the pow controller owns it; the mempool
        # producer's per-block System.Events poll rides the same manager.
        head_subscribers=(
            [mempool_stack.producer.on_new_block]
            if mempool_stack is not None
            else None
        ),
    )

    scheduler = WorkScheduler(
        core.miner_handles,
        on_pow_result=pow_controller.enqueue_pow_result,
        on_job_result=(
            mempool_stack.on_job_result if mempool_stack is not None else None
        ),
        provide_pow_context=pow_controller.provide_pow_context,
        on_worker_message=pow_controller.handle_worker_message,
        # A dead worker already sets the scheduler's own shutdown; mirror
        # it into the pow controller so FIRST_COMPLETED tears down cleanly.
        on_fatal=lambda _handle_id, _reason: pow_controller.shutdown(),
    )
    pow_controller.attach_scheduler(scheduler)
    if mempool_stack is not None:
        mempool_stack.attach_scheduler(scheduler)

    click.echo(
        f"  handles: {[h.miner_id for h in core.miner_handles]} "
        f"topology=0x{binding.chain_hash.hex()[:16]}... "
        f"mempool={'on' if mempool_stack is not None else 'off'}"
    )
    return pow_controller, scheduler, mempool_stack


def _announce_and_load(
    *,
    mempool_enabled: bool,
    validators: tuple,
    signer_key_path: str,
    faucet_url: Optional[str],
    rest_port: int,
    rest_host: str,
    miner_config: dict,
    node_name: Optional[str],
    node_log: Optional[str],
    submission_config: SubmissionConfig,
) -> object:
    """Print the startup banner and load the keystore.

    Attaches the optional rotating file log, loads the hybrid keystore
    (Guard A — raises ``click.ClickException`` when missing/unreadable), and
    echoes the resolved runtime config. The topology is no longer loaded here:
    the live miner pulls it from the chain's registered DefaultTopology after
    connecting (see :func:`_topology_from_chain`). Returns the keystore.
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
    click.echo(
        f"signer: {keystore.signer.ss58_address()} (hybrid) "
        f"mempool={'on' if mempool_enabled else 'off'}"
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
    return keystore


async def _run_concurrent_miner(
    *,
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
    mempool_enabled: bool = False,
    mempool_min_reward: int = 0,
) -> int:
    """Unified scheduler-stack entry for the cpu/gpu/qpu commands.

    Runs the linear startup sequence — announce + load identity
    (:func:`_announce_and_load`), connect + guards (incl. the non-fatal
    mempool Guard D+), build workers (:func:`_prepare_core`), build the
    single WorkScheduler stack (:func:`_build_scheduler_stack`) — then
    hands off to :func:`_orchestrate_controllers`. Every handle mines pow
    continuously; when *mempool_enabled*, matching mempool jobs preempt
    pow on the same workers and pow resumes after. Startup failures raise
    :class:`_MiningStartupError` (mapped to a CLI exit code below); the
    ``finally`` tears down whatever was constructed.
    """
    # --topology is tools-only. The live miner always binds to the chain's
    # registered DefaultTopology; accepting a locally-sourced graph here is what
    # let the miner's graph skew from the chain's, so reject it loudly.
    if topology_spec is not None:
        click.echo(
            f"--topology {topology_spec!r} is not valid for live mining: the "
            "miner binds to the chain's registered DefaultTopology and "
            "re-binds automatically when it changes. The flag exists only for "
            "standalone tools / isolated analysis run without a chain.",
            err=True,
        )
        return 2

    # Submission tuning (tip + retry bounds). Defaults reproduce pre-tip
    # behavior, so an old config or a caller that omits this is unchanged.
    submission_config = submission_config or SubmissionConfig()
    keystore = _announce_and_load(
        mempool_enabled=mempool_enabled,
        validators=validators,
        signer_key_path=signer_key_path,
        faucet_url=faucet_url,
        rest_port=rest_port,
        rest_host=rest_host,
        miner_config=miner_config,
        node_name=node_name,
        node_log=node_log,
        submission_config=submission_config,
    )

    # Declare all resources before the try so the finally block can always
    # clean up whatever was partially constructed on setup failure.
    core: Optional[MinerCore] = None
    pool: Optional[ValidatorPool] = None
    setup_client: Optional[SubstrateClient] = None
    try:
        pool = ValidatorPool(urls=tuple(validators))
        # Guards C/D/D+/E run once: a direct setup client connects, then
        # funded/registered/(solver-registered)/descriptor-filed are
        # verified before any mining. Guard D+ is non-fatal: it can only
        # flip mempool_enabled to False for this run.
        setup_client = await _connect_or_fail(tuple(validators))
        mempool_enabled = await _run_startup_guards(
            setup_client,
            keystore,
            faucet_url=faucet_url,
            node_name=node_name,
            public_host=public_host,
            public_port=public_port,
            miner_config=miner_config,
            mempool_enabled=mempool_enabled,
            miner_kind=miner_kind,
        )
        await setup_client.close()
        setup_client = None

        # Resolve telemetry port. The sibling process is unconditional;
        # any non-positive `rest_port` (legacy `-1` default, `0`, `None`)
        # collapses to 8086.
        telemetry_port = rest_port if rest_port and rest_port > 0 else 8086
        account_bytes = keystore.signer.account_id_bytes()

        # Bind-and-run loop: build the mining stack against the chain's current
        # DefaultTopology, run it, and rebuild from scratch when the controller
        # signals the chain changed its topology — no process restart needed.
        while True:
            setup_client = await _connect_or_fail(tuple(validators))
            topology, snapshot = await _topology_from_chain(
                setup_client,
                account_bytes=account_bytes,
                miner_config=miner_config,
            )
            click.echo(
                "topology (from chain): "
                f"0x{snapshot.topology_hash.hex()[:16]}... "
                f"({topology.num_nodes} nodes, {topology.num_edges} edges)"
            )
            core = _prepare_core(miner_kind, miner_config, topology)
            pow_controller = None
            scheduler = None
            mempool_stack = None
            try:
                (
                    pow_controller,
                    scheduler,
                    mempool_stack,
                ) = await _build_scheduler_stack(
                    client=setup_client,
                    pool=pool,
                    core=core,
                    keystore=keystore,
                    topology=topology,
                    miner_kind=miner_kind,
                    telemetry_port=telemetry_port,
                    submission_config=submission_config,
                    mempool_enabled=mempool_enabled,
                    mempool_min_reward=mempool_min_reward,
                )
                click.echo(
                    f"telemetry api: http://{rest_host}:{telemetry_port}"
                    "/api/v1/status (sibling process)"
                )
                # Controllers own the live parent-side connections (via the
                # pool) from here; close the direct setup client.
                await setup_client.close()
                setup_client = None

                rebind, exit_code = await _orchestrate_controllers(
                    pow_controller, scheduler, mempool_stack
                )
            finally:
                if pow_controller is not None:
                    click.echo(f"  pow stats:       {pow_controller.stats}")
                if scheduler is not None:
                    click.echo(f"  scheduler stats: {scheduler.stats}")
                if mempool_stack is not None:
                    click.echo(
                        f"  mempool stats:   producer={mempool_stack.producer.stats} "
                        f"submitter={mempool_stack.submitter.stats}"
                    )
                if core is not None:
                    core.close()
                    core = None

            if not rebind:
                return exit_code
            click.echo(
                "chain DefaultTopology changed; rebinding to the new chain "
                "topology and rebuilding the mining stack..."
            )

    except _MiningStartupError as exc:
        click.echo(exc.message, err=True)
        return exc.code
    finally:
        if setup_client is not None:
            await setup_client.close()
        if pool is not None:
            await pool.shutdown()
        if core is not None:
            core.close()


def _resolve_mempool_enabled(merged: dict, miner_kind: str) -> bool:
    """Effective `[miner] mempool` flag for this backend process.

    - Explicit TOML value always wins (`mempool = false` is preserved by
      the `is None` default machinery in :func:`_resolve_runtime_config`).
    - When the key is absent, cpu/gpu default ON and qpu (any vendor kind
      — `qpu`, `qpu_ibm`, ...) defaults OFF: QPU handles are never
      preempted and every dispatched job costs paid QPU time, so QPU
      participation is opt-in.
    - Env `QUIP_MEMPOOL=0` force-disables regardless of config: the T8
      supervisor owner-election sets it on every non-owner child so one
      account never races two solver registrations.
    """
    if os.environ.get("QUIP_MEMPOOL", "").strip() == "0":
        return False
    value = merged.get("mempool")
    if value is None:
        return not miner_kind.startswith("qpu")
    return bool(value)


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

    Bundles the validator / config / faucet / signer-key / topology / rest
    / identification options so each command declares only its
    backend-specific flags. Mempool participation is config-only
    (`[miner] mempool` / `mempool_min_reward` — see
    :func:`_resolve_mempool_enabled`). Apply *above*
    ``@click.pass_context`` so ``ctx`` stays the command's first
    positional argument.
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
        "--topology", "topology_spec", default=None,
        show_default=False, help=_TOPOLOGY_HELP,
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
    topology_spec: str,
    rest_port: Optional[int],
    rest_host: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    node_log: Optional[str],
) -> None:
    """Shared tail of the cpu/gpu/qpu commands; raises ``SystemExit``.

    Merges CLI overrides onto the already-parsed `raw` config, resolves the
    effective `[miner] mempool` flag for this backend, and hands off to
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
        mempool_enabled=_resolve_mempool_enabled(merged, miner_kind),
        mempool_min_reward=int(merged.get("mempool_min_reward") or 0),
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

    Every worker mines PoW (chain heads via QuantumPow.submit_proof)
    continuously. Mempool participation defaults ON for CPU: matching
    QuantumComputeMempool jobs preempt PoW on the same workers and PoW
    resumes after. Disable with `mempool = false` in the `[miner]` table;
    solver registration happens automatically at startup.
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
    topology_spec: str,
    rest_port: Optional[int],
    rest_host: Optional[str],
    node_name: Optional[str],
    public_host: Optional[str],
    public_port: Optional[int],
    node_log: Optional[str],
) -> None:
    """Run a GPU miner (CUDA / Metal / Modal) against a substrate chain.

    See `quip-miner cpu --help` for the PoW + mempool scheduling model;
    mempool participation defaults ON for GPU (disable with
    `mempool = false` in `[miner]`).
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
    See `quip-miner cpu --help` for the PoW + mempool scheduling model.
    Mempool participation defaults OFF for QPU (paid samples; jobs
    dispatch idle-only, never preempting in-flight QPU work) — opt in
    with `mempool = true` in the `[miner]` table.
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
        topology_spec=topology_spec,
        rest_port=rest_port,
        rest_host=rest_host,
        node_name=node_name,
        public_host=public_host,
        public_port=public_port,
        node_log=node_log,
    )


# --------------------------------------------------------------------------
# `quip-miner identify` — post a signed on-chain NodeDescriptor.
#
# Builds a v0.1-shaped `nodes.json` entry (schema=quip.node_descriptor.v1)
# from auto-detected hardware + optional --miner-config, canonicalizes to
# UTF-8 JSON for dry-run previews. Live submission uses
# MinerRegistry.set_descriptor. The signer's account is the canonical
# identity; dashboards/indexers read MinerRegistry.NodeDescriptors.
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
    """Post a signed NodeDescriptor for this miner.

    The descriptor carries node_name, optional public_host/public_port,
    advertised rpc_endpoints, auto_mine flag, runtime block (python /
    quip_version / docker), miners block (from --miner-config), and an
    auto-detected system_info block (skippable with --no-system-info).
    Submitted as `MinerRegistry.set_descriptor`; dry-run still prints the
    canonical JSON preview used by the local REST/dashboard surface.
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

    if dry_run:
        payload = to_canonical_json(descriptor)
        click.echo(payload.decode("utf-8"))
        click.echo(
            f"\n# account            : {keystore.signer.ss58_address()}", err=True
        )
        click.echo(f"# payload_size_bytes : {len(payload)}", err=True)
        return

    async def _do() -> int:
        client = await _connect_or_fail(tuple(merged["validators"]))
        try:
            # Post V2 (system_info + runtime) where the runtime supports it,
            # else V1 — probed from live metadata after connecting.
            schema_version = await client.descriptor_schema_version()
            try:
                call_params = descriptor_call_params(
                    descriptor, node_id=node_id, schema_version=schema_version,
                )
            except ValueError as exc:
                raise click.ClickException(f"descriptor-invalid {exc}") from exc
            receipt = await client.submit_extrinsic(
                "MinerRegistry",
                "set_descriptor",
                call_params,
                keystore.signer,
                wait_for="inblock",
            )
            if receipt.error:
                click.echo(f"set_descriptor failed: {receipt.error}", err=True)
                return 3
            click.echo("identify submitted")
            click.echo(f"  account            : {keystore.signer.ss58_address()}")
            click.echo("  call               : MinerRegistry.set_descriptor")
            click.echo(f"  extrinsic_hash     : {receipt.extrinsic_hash}")
            click.echo(f"  block_hash         : {receipt.block_hash}")
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
