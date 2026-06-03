# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Submit a ``System`` remark, preferring the eventful variant.

Dashboards key off the event emitted by ``System.remark_with_event``, so that
variant is used whenever the live runtime exposes it. Some metadata caches
answer ``has_call`` "yes" while the active runtime rejects the call at compose
time; when that happens the eventful variant's failure is reported via
``on_fallback`` and a plain ``System.remark`` is submitted instead.

Three call sites share this dance — the descriptor remark and the ``identify``
command in ``quip_cli``, and the controller's participation remark — so the
chain semantics live here rather than being copy-pasted into each.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from shared.signer import Signer
from substrate.client import SubstrateClient, WaitFor
from substrate.types import ExtrinsicReceipt


async def submit_remark(
    client: SubstrateClient,
    signer: Signer,
    payload: bytes,
    *,
    wait_for: WaitFor = "inblock",
    on_fallback: Optional[Callable[[Exception], None]] = None,
) -> Tuple[ExtrinsicReceipt, str]:
    """Submit ``payload`` as a ``System`` remark on ``client``.

    Prefers ``System.remark_with_event``; degrades to plain ``System.remark``
    when the eventful variant fails to compose against the live runtime,
    passing the triggering exception to ``on_fallback`` (if given) before the
    retry.

    Args:
        client: Connected substrate client to submit through.
        signer: Keystore signer for the extrinsic.
        payload: Raw remark bytes (already encoded by the caller).
        wait_for: Inclusion level to wait for before returning.
        on_fallback: Optional callback invoked with the eventful variant's
            exception when degrading to plain ``remark``; lets each caller log
            in its own voice without duplicating the fallback control flow.

    Returns:
        The receipt and the call name actually used (``"remark_with_event"``
        or ``"remark"``) so callers can report which path landed.

    Raises:
        Exception: Re-raised when the plain ``remark`` path itself fails.
    """
    prefer_event = await client.has_call("System", "remark_with_event")
    call_function = "remark_with_event" if prefer_event else "remark"
    try:
        receipt = await client.submit_extrinsic(
            "System", call_function, {"remark": payload}, signer,
            wait_for=wait_for,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to plain remark below
        if call_function == "remark":
            raise
        if on_fallback is not None:
            on_fallback(exc)
        receipt = await client.submit_extrinsic(
            "System", "remark", {"remark": payload}, signer,
            wait_for=wait_for,
        )
        call_function = "remark"
    return receipt, call_function
