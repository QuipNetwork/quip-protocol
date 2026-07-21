"""TOML config loader and CLI-merge helper for `quip-miner`.

Two slices of the TOML are read by separate loaders:

  * `load_miner_config(path)` → the `[miner]` table (connection,
    keystore, telemetry, identification — see schema below).
  * `load_backend_config(path)` → the v0.1-shape backend inventory
    tables that describe what hardware this rig runs (`[cpu]`,
    `[gpu]`, `[cuda.N]`, `[nvidia.N]`, `[metal]`, `[modal]`, `[qpu]`,
    `[dwave]`, `[ibm]`, `[braket]`, `[pasqal]`, `[ionq]`, `[origin]`).

Splitting the loaders preserves the original v0.1 file shape (where
backend tables sit at the top level next to `[global]`) while keeping
the v0.2 `[miner]` table as the canonical home for connection +
keystore + identification.

[miner] schema (keys actually read by the CLI today):

    [miner]
    validators  = ["ws://primary:9944", "ws://standby:9944"]
    signer_key  = "~/.quip-miner/signing.json"
    faucet_url  = "http://localhost:8087"   # optional
    rest_port   = 8086                      # optional, -1 disables
    rest_host   = "127.0.0.1"               # optional, 0.0.0.0 for containers

    # Identification metadata (used for auto-identify on startup;
    # also published in the NodeDescriptor remark dashboards consume).
    node_name   = "rig-01"                  # optional, defaults to hostname
    public_host = "miner.example.com"       # optional, ≤253 bytes
    public_port = 8086                      # optional, 1..65535
    log_level   = "INFO"                    # optional, DEBUG/INFO/WARNING/ERROR
    node_log    = "/var/log/quip-miner.log" # optional rotating file handler

    # Mempool participation is a PER-MINER property, set inside the
    # backend sections, not here: `[cpu] mempool = false`,
    # `[gpu] mempool = false` (also [metal]/[modal]), or a qpu vendor
    # section (`[dwave] mempool = true`). Defaults: cpu/gpu on, qpu off.
    # A `mempool` key in [miner] is rejected at load time.
    # `mempool_min_reward` drops orders paying less (0 = accept all).
    mempool_min_reward = 0                  # optional non-negative int

    # v0.1 compatibility aliases — `listen` → `rest_host`, `port` → `rest_port`.
    # These are convenience for v0.1 migrants; if both an alias and its
    # canonical key are present, the canonical key wins and the alias is
    # silently ignored (no warning — TOML is the operator's source of truth
    # and rejecting on duplicate keys would break legitimate "v0.1 base
    # config + v0.2 override" layering).
    listen      = "0.0.0.0"                 # alias for rest_host
    port        = 8086                      # alias for rest_port

CLI flags override TOML on a per-key basis for the `[miner]` table.
Empty CLI values (`None`, empty tuple/list) do *not* clobber TOML so
operators can mix `--config defaults.toml` with overrides for a single
run.

Backend tables follow a stricter rule: if the TOML defines a backend
section AND the operator passes a conflicting CLI flag (`--num-cpus`,
`--gpu-backend`, `--qpu-type`, `--daily-budget`), the CLI fails fast
with a `config-conflict` error. This avoids the v0.1 ambiguity where
operators couldn't tell whether the TOML or the flag drove a given
miner. Use one or the other.

`validators` is the canonical key for the failover-ordered list of
substrate validator WebSocket URLs. The TOML form must be a list of
strings; type errors fail fast at load time, before we touch the chain.
When the key is absent (and no --validator flag is given), the mining
commands fall back to ["ws://quip-validator:9944",
"ws://127.0.0.1:9944"] — see quip_cli._DEFAULT_VALIDATORS.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- exercised on 3.10 only
    import tomli as tomllib


class MinerConfigError(ValueError):
    """Operator-facing problem with a miner config file or merged settings."""


def load_miner_config(
    path: Optional[Path] = None, *, raw: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Read a TOML file and return its `[miner]` table.

    Returns `{}` when the file has no `[miner]` section so callers can
    treat "config present, miner section absent" the same as "no config
    file". Raises `MinerConfigError` for missing files, parse errors, or
    schema violations the CLI can surface verbatim.

    Pass `raw` (a pre-parsed mapping from :func:`load_toml`) to reuse a
    single parse instead of re-reading `path`.
    """
    data = _resolve_config(path, raw)
    miner = data.get("miner", {})
    if not isinstance(miner, Mapping):
        raise MinerConfigError(
            f"miner config: [miner] must be a table, got {type(miner).__name__}"
        )
    miner_dict = dict(miner)
    _validate_validators_field(miner_dict.get("validators"))
    if "mempool" in miner_dict:
        raise MinerConfigError(
            "miner config: `mempool` is a per-miner property — move it "
            "from [miner] into the backend section it applies to "
            "([cpu], [gpu], [metal], [modal], or a qpu vendor section "
            "like [dwave]). Defaults: cpu/gpu on, qpu off."
        )
    _validate_mempool_min_reward_field(miner_dict.get("mempool_min_reward"))
    _apply_v01_aliases(miner_dict)
    return miner_dict


@dataclass(frozen=True)
class SubmissionConfig:
    """Tuning for the ``submit_proof`` extrinsic, from the ``[submission]`` table.

    ``tip_plancks`` is the ``ChargeTransactionPayment`` tip (chain base unit)
    threaded into the signed extrinsic to raise its transaction-pool
    priority — the round goes to the lowest-energy proof *within a block*,
    so inclusion in the right block is the race. ``max_retries`` /
    ``retry_backoff_ms`` bound the resilient submit loop
    (:func:`substrate.submitter.submit_with_retry`): ``max_retries`` is the
    number of *additional* attempts after the first; backoff is linear per
    attempt. Defaults are no-tip + 3 retries + 250 ms so omitting the
    section reproduces today's behavior.
    """

    tip_plancks: int = 0
    max_retries: int = 3
    retry_backoff_ms: int = 250


def load_submission_config(
    path: Optional[Path] = None, *, raw: Optional[Mapping[str, Any]] = None
) -> SubmissionConfig:
    """Read the ``[submission]`` table into a :class:`SubmissionConfig`.

    Returns defaults when the file has no ``[submission]`` section, so a
    config that predates this feature behaves exactly as before. Raises
    :class:`MinerConfigError` for a non-table section or out-of-range
    values — these reach the submit path, so a malformed knob should fail
    at load time, not at the first proof.

    Pass `raw` (a pre-parsed mapping from :func:`load_toml`) to reuse a
    single parse instead of re-reading `path`.
    """
    data = _resolve_config(path, raw)
    section = data.get("submission", {})
    if not isinstance(section, Mapping):
        raise MinerConfigError(
            "miner config: [submission] must be a table, got "
            f"{type(section).__name__}"
        )
    return _parse_submission_table(section)


# The chain encodes the tip as a `Compact<Balance>` where Balance is u128.
# A tip above this encodes fine locally but the runtime rejects it — catch
# the fail-slow at config-load time instead.
_U128_MAX: int = (1 << 128) - 1


def _parse_submission_table(section: Mapping[str, Any]) -> SubmissionConfig:
    """Validate + coerce a ``[submission]`` mapping to a config dataclass."""
    defaults = SubmissionConfig()
    tip = _require_non_negative_int(
        section, "tip_plancks", defaults.tip_plancks, maximum=_U128_MAX
    )
    max_retries = _require_non_negative_int(
        section, "max_retries", defaults.max_retries
    )
    backoff = _require_non_negative_int(
        section, "retry_backoff_ms", defaults.retry_backoff_ms
    )
    return SubmissionConfig(
        tip_plancks=tip,
        max_retries=max_retries,
        retry_backoff_ms=backoff,
    )


def _require_non_negative_int(
    section: Mapping[str, Any],
    key: str,
    default: int,
    *,
    maximum: Optional[int] = None,
) -> int:
    """Read ``key`` as a non-negative int, defaulting when absent.

    When ``maximum`` is given, values above it raise — used to keep
    ``tip_plancks`` within the chain's u128 ``Balance`` range so an
    over-large tip fails at load time rather than at submit time.
    """
    if key not in section:
        return default
    value = section[key]
    # bool is an int subclass; reject it so `tip_plancks = true` fails loud.
    if isinstance(value, bool) or not isinstance(value, int):
        raise MinerConfigError(
            f"miner config: [submission].{key} must be a non-negative "
            f"integer, got {value!r}"
        )
    if value < 0:
        raise MinerConfigError(
            f"miner config: [submission].{key} must be non-negative, got {value}"
        )
    if maximum is not None and value > maximum:
        raise MinerConfigError(
            f"miner config: [submission].{key} must be <= {maximum} "
            f"(u128 max), got {value}"
        )
    return value


# Names of the v0.1 top-level tables that describe miner hardware
# inventory. The grouping is used both for backend discovery in
# `load_backend_config` and for CLI conflict detection in
# `present_backend_groups`.
CPU_BACKEND_SECTIONS: tuple[str, ...] = ("cpu",)
GPU_BACKEND_SECTIONS: tuple[str, ...] = ("gpu", "cuda", "nvidia", "metal", "modal")
QPU_BACKEND_SECTIONS: tuple[str, ...] = (
    "qpu",
    "dwave",
    "ibm",
    "braket",
    "pasqal",
    "ionq",
    "origin",
)
ALL_BACKEND_SECTIONS: tuple[str, ...] = (
    *CPU_BACKEND_SECTIONS,
    *GPU_BACKEND_SECTIONS,
    *QPU_BACKEND_SECTIONS,
)


def load_backend_config(
    path: Optional[Path] = None, *, raw: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Read a TOML file and return the v0.1-shape backend inventory tables.

    Returns a dict containing only the recognised hardware-inventory
    sections (`[cpu]`, `[gpu]`, `[cuda.N]`, `[nvidia.N]`, `[metal]`,
    `[modal]`, `[qpu]`, `[dwave]`, `[ibm]`, `[braket]`, `[pasqal]`,
    `[ionq]`, `[origin]`). Values are returned verbatim — the consumer
    (`MinerCore._initialize_miners`, via `_build_gpu_specs` /
    `_build_qpu_specs`) already handles the per-device sub-table /
    array-of-tables shapes the v0.1 schema supports.

    Returns `{}` when the file has no recognised backend sections.
    Raises `MinerConfigError` for missing files or parse errors.

    Pass `raw` (a pre-parsed mapping from :func:`load_toml`) to reuse a
    single parse instead of re-reading `path`.
    """
    data = _resolve_config(path, raw)
    out: dict[str, Any] = {}
    for section in ALL_BACKEND_SECTIONS:
        value = data.get(section)
        if value is None:
            continue
        # Allow both `[gpu]` (mapping) and `[[cuda]]` (list of tables) —
        # `_normalize_gpu_config` / `_normalize_qpu_config` accept both.
        if not isinstance(value, (Mapping, list)):
            raise MinerConfigError(
                f"miner config: [{section}] must be a table or array of tables, "
                f"got {type(value).__name__}"
            )
        out[section] = value
    return out


def present_backend_groups(backends: Mapping[str, Any]) -> dict[str, list[str]]:
    """Bucket the section names in `backends` by hardware group.

    Returns `{"cpu": [...], "gpu": [...], "qpu": [...]}` listing the
    section names present in each group. Empty list when the group has
    no sections. Callers use this for conflict detection against the
    cpu/gpu/qpu CLI flags.
    """
    return {
        "cpu": [s for s in CPU_BACKEND_SECTIONS if s in backends],
        "gpu": [s for s in GPU_BACKEND_SECTIONS if s in backends],
        "qpu": [s for s in QPU_BACKEND_SECTIONS if s in backends],
    }


MODE_NAMES: tuple[str, ...] = ("cpu", "gpu", "qpu")


# Sections that may carry the per-miner `mempool` key: the flat tables
# of each group. The [cuda.N]/[nvidia.N] device maps are excluded — the
# key is group-level (one solver type per substrate account), never
# per-device.
_MEMPOOL_KEY_SECTIONS: dict[str, tuple[str, ...]] = {
    "cpu": CPU_BACKEND_SECTIONS,
    "gpu": ("gpu", "metal", "modal"),
    "qpu": QPU_BACKEND_SECTIONS,
}


def group_mempool_value(
    backends: Mapping[str, Any], group: str
) -> Optional[bool]:
    """The explicit per-miner `mempool` value for a backend group.

    Reads the key from the group's flat sections (`[cpu]`, the `[gpu]`
    defaults table, `[metal]`/`[modal]`, or a qpu vendor section like
    `[dwave]`). Returns ``None`` when no section sets it. Raises
    :class:`MinerConfigError` on a non-bool value (the TOML string
    ``"false"`` is truthy and would silently enable participation) or
    when two sections of the same group disagree.
    """
    found: dict[str, bool] = {}
    for section in _MEMPOOL_KEY_SECTIONS[group]:
        table = backends.get(section)
        if not isinstance(table, Mapping) or "mempool" not in table:
            continue
        value = table["mempool"]
        if not isinstance(value, bool):
            raise MinerConfigError(
                f"miner config: [{section}].mempool must be a TOML "
                f"boolean (true/false, unquoted), got "
                f"{type(value).__name__} ({value!r})"
            )
        found[section] = value
    if len(set(found.values())) > 1:
        detail = ", ".join(f"[{s}]={v}" for s, v in found.items())
        raise MinerConfigError(
            f"miner config: conflicting `mempool` values within the "
            f"{group} group ({detail}); the key is group-level — set it "
            f"once"
        )
    return next(iter(found.values()), None)


def mempool_owner_group(backends: Mapping[str, Any]) -> Optional[str]:
    """The one backend group whose child participates in the mempool.

    A pure function of the config's backend sections — derived from the
    same TOML by every process, with no out-of-band transport, so
    supervised, direct-subcommand, and ``--mode``-narrowed runs all
    agree. One substrate account can only register one solver type on
    chain, so every non-owner child resolves mempool off.

    Election, over the configured groups' per-miner `mempool` keys
    (see :func:`group_mempool_value`):

    1. An explicit ``mempool = true`` states intent and outranks groups
       that are merely default-on (this is how a lone
       ``[dwave] mempool = true`` owns the mempool). Canonical
       cpu,gpu,qpu order breaks ties.
    2. Otherwise the first default-on group owns — cpu/gpu default on,
       qpu defaults off (paid samples are opt-in).
    3. ``None`` when nothing is enabled (explicit opt-outs, qpu-only,
       or empty backends).
    """
    groups = present_backend_groups(backends)
    values = {
        g: group_mempool_value(backends, g)
        for g in MODE_NAMES
        if groups[g]
    }
    for g in MODE_NAMES:  # explicit opt-ins first
        if values.get(g) is True:
            return g
    for g in MODE_NAMES:  # then default-on groups (qpu is default-off)
        if g != "qpu" and g in values and values[g] is None:
            return g
    return None


class ModeResolutionError(MinerConfigError):
    """Operator-actionable problem in `resolve_mode`.

    Subclasses `MinerConfigError` so the CLI's existing error-formatting
    path handles it the same way as load/validate errors. The `code`
    attribute is the machine-parseable kebab-case identifier the
    entrypoint scripts can grep for.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code


def resolve_modes(
    backends: Mapping[str, Any],
    *,
    default: Optional[str] = None,
    image_supports: Optional[Sequence[str]] = None,
) -> list[str]:
    """Pick the quip-miner subcommands (subset of `cpu` / `gpu` / `qpu`)
    that should run for the given config.

    Resolution rules, in order:

      1. If `backends` declares one or more active groups → return them
         in canonical order (cpu, gpu, qpu) — one mode per group.
         Each becomes a separate `quip-miner <mode>` process under the
         entrypoint supervisor.
      2. If `backends` is empty and `default` is provided → return
         `[default]` (a single fallback mode).
      3. Otherwise → raise `no-mode-resolvable`.

    `image_supports`, when provided, restricts the acceptable answers
    to that subset of `MODE_NAMES`. Any resolved mode outside the
    supported list raises `unsupported-mode` — the image can't run
    hardware it doesn't have libraries for.

    Returns a list of one or more modes (always non-empty when no
    error). Raises `ModeResolutionError` with a kebab-case `code` on
    any rule failure.
    """
    if image_supports is not None:
        supports = tuple(image_supports)
        unknown = [m for m in supports if m not in MODE_NAMES]
        if unknown:
            raise ModeResolutionError(
                "bad-image-supports",
                f"unknown modes in image-supports={list(supports)}; "
                f"expected subset of {list(MODE_NAMES)}",
            )
    else:
        supports = MODE_NAMES

    if default is not None and default not in MODE_NAMES:
        raise ModeResolutionError(
            "bad-default",
            f"default={default!r} is not a valid mode; "
            f"expected one of {list(MODE_NAMES)}",
        )

    groups = present_backend_groups(backends)
    active = [g for g in MODE_NAMES if groups[g]]

    if active:
        unsupported = [g for g in active if g not in supports]
        if unsupported:
            details = ", ".join(
                f"{g}={groups[g]}" for g in unsupported
            )
            raise ModeResolutionError(
                "unsupported-mode",
                f"config requires mode(s) the image cannot run: {details}; "
                f"image supports {list(supports)}. Either run a different "
                f"image (gpu image for [cuda.N], any image for [dwave]) or "
                f"remove the offending section from config.",
            )
        return active

    # No active groups in config — fall back to default.
    if default is None:
        raise ModeResolutionError(
            "no-mode-resolvable",
            "config has no backend sections and no --default given; "
            "add one of [cpu]/[gpu]/[cuda.N]/[metal]/[modal]/[qpu]/"
            "[dwave]/[ibm]/[braket]/[pasqal]/[ionq]/[origin] to the "
            "config, or pass --default {cpu,gpu,qpu}.",
        )
    if default not in supports:
        raise ModeResolutionError(
            "unsupported-mode",
            f"default={default} is not in image-supports={list(supports)}. "
            f"Pass --default to a supported mode or add a backend section "
            f"to config.",
        )
    return [default]


def resolve_mode(
    backends: Mapping[str, Any],
    *,
    default: Optional[str] = None,
    image_supports: Optional[Sequence[str]] = None,
) -> str:
    """Single-mode variant of `resolve_modes` — convenience for callers
    that won't supervise multiple processes.

    Errors with `multi-backend-not-single-mode` when the config
    resolves to more than one mode. The supervisor entrypoint uses
    `resolve_modes` instead; this is for one-shot tooling and tests.
    """
    modes = resolve_modes(
        backends,
        default=default,
        image_supports=image_supports,
    )
    if len(modes) > 1:
        raise ModeResolutionError(
            "multi-backend-not-single-mode",
            f"config resolves to multiple modes ({modes}); caller wanted "
            f"a single mode. Use `resolve_modes` and supervise one "
            f"process per mode.",
        )
    return modes[0]


def load_toml(path: Path) -> dict[str, Any]:
    """Read + parse a TOML file. Shared by every ``load_*`` entry point.

    Public so a caller that needs several views of the same file (e.g. the
    cpu/gpu/qpu CLI, which wants the ``[miner]``, backend, and ``[submission]``
    tables) can parse once and pass the result back in via the ``raw=``
    parameter rather than re-reading + re-parsing the file three times.
    """
    if not path.exists():
        raise MinerConfigError(f"miner config not found: {path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise MinerConfigError(
            f"miner config parse failed ({path}): {exc}"
        ) from exc


def _resolve_config(
    path: Optional[Path], raw: Optional[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Return the parsed config: ``raw`` when supplied, else parse ``path``.

    Lets the ``load_*`` functions accept a pre-parsed mapping so a caller can
    parse a file once and feed it to all of them.
    """
    if raw is not None:
        return raw
    if path is None:
        raise MinerConfigError("config load requires a path or pre-parsed raw")
    return load_toml(path)


def merge_config(
    toml_data: Mapping[str, Any], cli_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay CLI values onto TOML defaults.

    A CLI value "counts" when it is not `None` and not an empty
    tuple/list. This matches click's behavior for repeatable options
    (`multiple=True` produces `()` when nothing was passed) and for
    optional flags that default to `None`.

    Tuples coming from click `multiple=True` are materialized to lists
    so downstream code sees the same type whether the value came from
    TOML or CLI.
    """
    out: dict[str, Any] = dict(toml_data)
    for key, value in cli_kwargs.items():
        if value is None:
            continue
        if isinstance(value, (tuple, list)) and len(value) == 0:
            continue
        if isinstance(value, tuple):
            value = list(value)
        out[key] = value
    return out


def validate_merged(merged: Mapping[str, Any]) -> None:
    """Sanity-check a merged config before handing it to `_run_miner`.

    Surfaces missing required keys (`validators`, `signer_key`) and
    empty-list validators with `MinerConfigError`. Caller catches and
    re-raises as `click.UsageError` / `click.ClickException` for stderr
    formatting.
    """
    validators = merged.get("validators")
    if validators is None:
        raise MinerConfigError(
            "no validator URLs configured. Pass --validator URL (repeatable) "
            "or add `validators = [...]` to your --config TOML"
        )
    if not isinstance(validators, list) or not validators:
        raise MinerConfigError(
            "at least one validator URL is required. Pass --validator URL "
            "(repeatable) or add `validators = [\"ws://...\"]` to --config"
        )
    signer_key = merged.get("signer_key")
    if not signer_key:
        raise MinerConfigError(
            "signer_key is required. Pass --signer-key PATH or add "
            "`signer_key = \"~/.quip-miner/signing.json\"` to --config"
        )


_V01_ALIASES: dict[str, str] = {
    # v0.1 [global].listen / .port served the QUIC server. v0.2 has only
    # the telemetry REST API as a listen surface, so map both to the rest_*
    # canonical keys. If a config sets both `listen` and `rest_host` the
    # canonical wins — the operator can stage v0.2 settings inside a v0.1
    # config without breaking the file for v0.1 users.
    "listen": "rest_host",
    "port": "rest_port",
}


def _apply_v01_aliases(miner_dict: dict[str, Any]) -> None:
    """Promote v0.1 alias keys to their v0.2 canonical names in-place.

    Canonical-wins-on-collision: if both keys are present, the alias is
    silently dropped. No warning — see the [miner] schema docstring for
    rationale.
    """
    for alias, canonical in _V01_ALIASES.items():
        if alias not in miner_dict:
            continue
        if canonical not in miner_dict:
            miner_dict[canonical] = miner_dict[alias]
        del miner_dict[alias]


def _validate_mempool_min_reward_field(value: Any) -> None:
    """Reject a non-int or negative `mempool_min_reward` at load time."""
    if value is None:
        return
    # bool is an int subclass; reject so `mempool_min_reward = true` fails loud.
    if isinstance(value, bool) or not isinstance(value, int):
        raise MinerConfigError(
            f"miner config: [miner].mempool_min_reward must be a "
            f"non-negative integer, got {type(value).__name__} ({value!r})"
        )
    if value < 0:
        raise MinerConfigError(
            f"miner config: [miner].mempool_min_reward must be non-negative, "
            f"got {value}"
        )


def _validate_validators_field(value: Any) -> None:
    """Reject non-string entries early so the operator sees the typo
    before we try to open a websocket to an integer."""
    if value is None:
        return
    if not isinstance(value, list):
        raise MinerConfigError(
            f"miner config: `validators` must be a list, got {type(value).__name__}"
        )
    for i, entry in enumerate(value):
        if not isinstance(entry, str):
            raise MinerConfigError(
                f"miner config: validators[{i}] must be a string, "
                f"got {type(entry).__name__} ({entry!r})"
            )


__all__ = [
    "ALL_BACKEND_SECTIONS",
    "CPU_BACKEND_SECTIONS",
    "GPU_BACKEND_SECTIONS",
    "MODE_NAMES",
    "MinerConfigError",
    "ModeResolutionError",
    "QPU_BACKEND_SECTIONS",
    "group_mempool_value",
    "load_backend_config",
    "load_miner_config",
    "mempool_owner_group",
    "merge_config",
    "present_backend_groups",
    "resolve_mode",
    "resolve_modes",
    "validate_merged",
]
