"""TOML config loader and CLI-merge helper for `quip-miner`.

Schema (keys actually read by the CLI today):

    [miner]
    validators = ["ws://primary:9944", "ws://standby:9944"]
    signer_key = "~/.quip-miner/signing.json"
    faucet_url = "http://localhost:8087"   # optional

Any other keys in `[miner]` (or nested `[miner.cpu]` / `[miner.gpu]` /
`[miner.qpu]` subtables) are parsed but ignored — CLI flags are the
only driver for `topology`, `rest_port`, and per-backend settings in
v0.2. Don't add keys to this docstring without wiring them through
`_resolve_runtime_config` first.

CLI flags override TOML on a per-key basis. Empty CLI values
(`None`, empty tuple/list) do *not* clobber TOML so operators can mix
`--config defaults.toml` with overrides for a single run.

`validators` is the canonical key for the failover-ordered list of
substrate validator WebSocket URLs. The TOML form must be a list of
strings; type errors fail fast at load time, before we touch the chain.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- exercised on 3.10 only
    import tomli as tomllib


class MinerConfigError(ValueError):
    """Operator-facing problem with a miner config file or merged settings."""


def load_miner_config(path: Path) -> dict[str, Any]:
    """Read a TOML file and return its `[miner]` table.

    Returns `{}` when the file has no `[miner]` section so callers can
    treat "config present, miner section absent" the same as "no config
    file". Raises `MinerConfigError` for missing files, parse errors, or
    schema violations the CLI can surface verbatim.
    """
    if not path.exists():
        raise MinerConfigError(f"miner config not found: {path}")
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise MinerConfigError(f"miner config parse failed ({path}): {exc}") from exc
    miner = raw.get("miner", {})
    if not isinstance(miner, Mapping):
        raise MinerConfigError(
            f"miner config: [miner] must be a table, got {type(miner).__name__}"
        )
    miner_dict = dict(miner)
    _validate_validators_field(miner_dict.get("validators"))
    return miner_dict


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
    "MinerConfigError",
    "load_miner_config",
    "merge_config",
    "validate_merged",
]
