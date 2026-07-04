"""Unit tests for `substrate.miner_bootstrap`.

These don't need a live chain — they stub `SubstrateClient` and exercise the
state machine directly. The highest-value coverage is the `_assert_dev_chain`
guard (without it, `--seed-chain` can sudo a production chain) and the
`_build_seed_topology` invariants the Rust pallet's
`validate_topology_consistency` relies on.
"""
from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

import substrate.miner_bootstrap as mb
from substrate.miner_bootstrap import (
    DEV_CHAIN_PREFIXES,
    BootstrapConfig,
    _assert_dev_chain,
    _build_seed_topology,
)


# ---------------------------------------------------------------------------
# _assert_dev_chain
# ---------------------------------------------------------------------------


class _StubClient:
    """Minimal stand-in for SubstrateClient that just carries a chain name."""

    def __init__(self, chain_name: str) -> None:
        self._iface = SimpleNamespace(chain=chain_name)

    async def _run(self, fn):
        # Bootstrap's `_assert_dev_chain` calls `client._run(lambda: ...)`.
        # The real client offloads to an executor; the stub just calls
        # synchronously since the underlying op is property access.
        return fn()


@pytest.mark.parametrize("name", [
    "Development",
    "Local Testnet",
    "Local Testnet (3 Validators)",
    "quip-local",
    "quip-local-dev",
])
async def test_assert_dev_chain_accepts_dev_prefixes(name):
    # Must not raise — the docstring of bootstrap promises these prefixes.
    await _assert_dev_chain(_StubClient(name))


@pytest.mark.parametrize("name", [
    "Quip",
    "Quip Production",
    "Polkadot",
    "Kusama",
    "Westend",
    "",
])
async def test_assert_dev_chain_rejects_non_dev_chains(name):
    with pytest.raises(RuntimeError, match="non-dev chain"):
        await _assert_dev_chain(_StubClient(name))


def test_dev_chain_prefixes_constant_matches_faucet():
    # The bootstrap and the faucet bot intentionally keep duplicate copies
    # of this list (so the faucet stays standalone). Pin the values here so
    # a drift on either side gets flagged by tests on both sides.
    assert DEV_CHAIN_PREFIXES == ("Development", "Local Testnet", "quip-local")


# ---------------------------------------------------------------------------
# _build_seed_topology
# ---------------------------------------------------------------------------


def test_build_seed_topology_returns_unique_sorted_ids():
    # Whatever labeling the helper picks (dense 0..n-1 in Phase 2, raw
    # zephyr int-labels in Phase 4 after the relabel-bug fix), it must
    # emit unique ascending non-negative ints.
    nodes, edges = _build_seed_topology((2, 2))
    assert len(nodes) > 0
    assert nodes == sorted(nodes)
    assert len(set(nodes)) == len(nodes)
    assert all(isinstance(n, int) and n >= 0 for n in nodes)


def test_build_seed_topology_edges_are_canonical_and_sorted():
    nodes, edges = _build_seed_topology((2, 2))
    # u < v on every edge
    assert all(u < v for u, v in edges), "edges must satisfy u < v"
    # globally sorted
    assert edges == sorted(edges), "edges must be sorted"
    # no self-loops, no duplicates
    assert len({(u, v) for u, v in edges}) == len(edges)
    assert all(u != v for u, v in edges)


def test_build_seed_topology_is_deterministic():
    a_nodes, a_edges = _build_seed_topology((2, 2))
    b_nodes, b_edges = _build_seed_topology((2, 2))
    assert a_nodes == b_nodes
    assert a_edges == b_edges


def test_build_seed_topology_edges_reference_only_emitted_nodes():
    nodes, edges = _build_seed_topology((2, 2))
    node_set = set(nodes)
    for u, v in edges:
        assert u in node_set and v in node_set


# ---------------------------------------------------------------------------
# Faucet funding retry (ensure_funded / _post_faucet / _ensure_registered)
# ---------------------------------------------------------------------------


async def _no_sleep(_seconds):
    """Drop-in for asyncio.sleep so retry tests don't wait in real time."""
    return None


class _BalanceClient:
    """Stub client returning a scripted sequence of balances (holds the last).

    Also answers ``get_sync_state`` so the faucet/node-disagreement path can
    tell a lagging node (``syncing=True``, keep waiting) from a fully-synced
    one on a different chain (``syncing=False``, fast-fail). ``sync_error``
    forces the probe to raise, exercising the fail-open-to-retry guard.
    """

    def __init__(
        self, balances, *, syncing=False, current_block=0,
        highest_block=0, sync_error=None,
    ):
        self._balances = list(balances)
        self.query_count = 0
        self._syncing = syncing
        self._current_block = current_block
        self._highest_block = highest_block
        self._sync_error = sync_error
        self.sync_probe_count = 0

    async def query_balance(self, _account):
        idx = min(self.query_count, len(self._balances) - 1)
        self.query_count += 1
        return self._balances[idx]

    async def get_sync_state(self):
        self.sync_probe_count += 1
        if self._sync_error is not None:
            raise self._sync_error
        return {
            "is_syncing": self._syncing,
            "peers": 1,
            "current_block": self._current_block,
            "highest_block": self._highest_block,
            "starting_block": 0,
        }


def _stub_keystore():
    signer = SimpleNamespace(
        account_id_bytes=lambda: b"\x11" * 32,
        ss58_address=lambda: "5StubAddr",
    )
    return SimpleNamespace(signer=signer)


def _faucet_config(budget=300.0, faucet_url="http://faucet.test"):
    return BootstrapConfig(
        validators=("ws://x",),
        signer_key_path=Path("/tmp/k.json"),
        faucet_url=faucet_url,
        min_balance_plancks=1000,
        faucet_top_up_plancks=5000,
        faucet_timeout_seconds=budget,
    )


async def test_ensure_funded_returns_when_already_funded(monkeypatch):
    posted = []
    monkeypatch.setattr(mb, "_post_faucet", lambda *a, **k: posted.append(1))
    bal = await mb.ensure_funded(_BalanceClient([2000]), _stub_keystore(), _faucet_config())
    assert bal == 2000
    assert posted == [], "faucet must not be hit when already funded"


async def test_ensure_funded_no_faucet_url_raises():
    client = _BalanceClient([0])
    with pytest.raises(RuntimeError, match="no --faucet-url"):
        await mb.ensure_funded(client, _stub_keystore(), _faucet_config(faucet_url=None))


async def test_ensure_funded_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)
    posts = []

    def fake_post(_url, *, dest_hex, amount):
        posts.append(amount)
        if len(posts) < 3:
            raise mb.FaucetTransientError(f"faucet returned 502 (try {len(posts)})")
        return {"ok": True}

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    # Balances are consumed by the initial already-funded check plus one read
    # per attempt; it settles only on the query after the 3rd (successful) POST.
    bal = await mb.ensure_funded(_BalanceClient([0, 0, 0, 5000]), _stub_keystore(), _faucet_config())
    assert bal == 5000
    assert len(posts) == 3, "should retry the two transient failures"


async def test_ensure_funded_permanent_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)
    posts = []

    def fake_post(_url, *, dest_hex, amount):
        posts.append(1)
        raise mb.FaucetPermanentError("faucet returned 400: bad dest")

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    with pytest.raises(mb.FaucetPermanentError):
        await mb.ensure_funded(_BalanceClient([0]), _stub_keystore(), _faucet_config())
    assert len(posts) == 1, "permanent errors must fail fast, not retry"


async def test_ensure_funded_already_funded_reconciles_via_balance(monkeypatch):
    # 403 "already funded" must NOT crash: the re-read balance is the truth and
    # shows we cleared min_balance, so funding succeeds on the next loop. The
    # node reads 0 then 5000 across the two loops — a lagging node catching up,
    # so the sync probe reports syncing=True and the loop keeps waiting.
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)
    posts = []

    def fake_post(_url, *, dest_hex, amount):
        posts.append(1)
        raise mb.FaucetAlreadyFunded(5000, "faucet returned 403: already funded")

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    # Initial check sees 0 (below min), then the post-403 re-read sees 5000.
    client = _BalanceClient([0, 5000], syncing=True)
    bal = await mb.ensure_funded(client, _stub_keystore(), _faucet_config())
    assert bal == 5000


async def test_ensure_funded_faucet_funded_but_node_synced_fast_fails(monkeypatch):
    # The observed production bug: faucet reports the account holds free=20000
    # (>= min=1000) but our fully-synced node reports 0 for the SAME account and
    # field. Retrying can't reconcile two synced reads that disagree — the miner
    # and faucet are on different chains. Fail fast, don't spin the budget.
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)
    posts = []

    def fake_post(_url, *, dest_hex, amount):
        posts.append(1)
        raise mb.FaucetAlreadyFunded(20000, "faucet returned 403: already funded")

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    client = _BalanceClient([0], syncing=False)
    with pytest.raises(RuntimeError, match="different chain"):
        await mb.ensure_funded(client, _stub_keystore(), _faucet_config())
    assert len(posts) == 1, "synced-node contradiction must fail fast, not retry"


async def test_ensure_funded_faucet_funded_but_node_syncing_waits(monkeypatch):
    # Same faucet/node disagreement, but the node is still syncing and catches
    # up to 20000 on the next loop. Must keep retrying (not fast-fail) and
    # succeed once the local balance clears min.
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)

    def fake_post(_url, *, dest_hex, amount):
        raise mb.FaucetAlreadyFunded(20000, "faucet returned 403: already funded")

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    # Pre-loop check reads 0, loop attempt-1 reads 0 (probes, still syncing),
    # attempt-2 reads 20000 (node caught up → success).
    client = _BalanceClient([0, 0, 20000], syncing=True, current_block=5, highest_block=9)
    bal = await mb.ensure_funded(client, _stub_keystore(), _faucet_config())
    assert bal == 20000
    assert client.sync_probe_count >= 1, "disagreement must probe sync state"


async def test_ensure_funded_faucet_funded_probe_error_keeps_retrying(monkeypatch):
    # If the sync-state probe itself fails we can't prove the node is synced,
    # so fail open to the retry budget rather than crashing or false fast-fail.
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)

    def fake_post(_url, *, dest_hex, amount):
        raise mb.FaucetAlreadyFunded(20000, "faucet returned 403: already funded")

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    client = _BalanceClient([0], sync_error=RuntimeError("probe timeout"))
    with pytest.raises(RuntimeError, match="did not fund within"):
        await mb.ensure_funded(client, _stub_keystore(), _faucet_config(budget=10.0))


async def test_ensure_funded_already_funded_below_min_bounds_not_crashes(monkeypatch):
    # max_funded < min_balance: faucet keeps refusing and balance never clears.
    # The funding budget must bound this into a single RuntimeError, not an
    # immediate crash on the first 403.
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)

    def fake_post(_url, *, dest_hex, amount):
        raise mb.FaucetAlreadyFunded(500, "faucet returned 403: already funded")

    monkeypatch.setattr(mb, "_post_faucet", fake_post)
    with pytest.raises(RuntimeError, match="did not fund within"):
        await mb.ensure_funded(_BalanceClient([500]), _stub_keystore(), _faucet_config(budget=10.0))


async def test_ensure_funded_gives_up_after_budget(monkeypatch):
    monkeypatch.setattr(mb.asyncio, "sleep", _no_sleep)

    def always_transient(_url, *, dest_hex, amount):
        raise mb.FaucetTransientError("faucet returned 503")

    monkeypatch.setattr(mb, "_post_faucet", always_transient)
    # Balance never settles; the planned-backoff budget bounds the loop.
    with pytest.raises(RuntimeError, match="did not fund within"):
        await mb.ensure_funded(_BalanceClient([0]), _stub_keystore(), _faucet_config(budget=10.0))


def _http_error(code, body=b'{"error": "x"}'):
    return urllib.error.HTTPError(
        url="http://f/request", code=code, msg="m", hdrs=None,
        fp=io.BytesIO(body),
    )


def test_post_faucet_classifies_5xx_as_transient(monkeypatch):
    monkeypatch.setattr(
        mb.urllib.request, "urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(_http_error(502)),
    )
    with pytest.raises(mb.FaucetTransientError):
        mb._post_faucet("http://f", dest_hex="0x00", amount=1)


def test_post_faucet_classifies_429_as_transient(monkeypatch):
    monkeypatch.setattr(
        mb.urllib.request, "urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(_http_error(429)),
    )
    with pytest.raises(mb.FaucetTransientError):
        mb._post_faucet("http://f", dest_hex="0x00", amount=1)


def test_post_faucet_classifies_400_as_permanent(monkeypatch):
    # 400 (malformed request) is the only fast-fail: retrying re-sends the
    # same bad request, so the caller should not loop on it.
    monkeypatch.setattr(
        mb.urllib.request, "urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(_http_error(400)),
    )
    with pytest.raises(mb.FaucetPermanentError):
        mb._post_faucet("http://f", dest_hex="0x00", amount=1)


def test_post_faucet_unexpected_4xx_is_transient(monkeypatch):
    # An unexpected 4xx (e.g. 409) is NOT fast-failed — the default is to
    # retry within the budget rather than kill a long-running miner.
    monkeypatch.setattr(
        mb.urllib.request, "urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(_http_error(409)),
    )
    with pytest.raises(mb.FaucetTransientError):
        mb._post_faucet("http://f", dest_hex="0x00", amount=1)


def test_post_faucet_403_already_funded_carries_free_balance(monkeypatch):
    body = b'{"error": "destination already funded", "free_balance_plancks": 42}'
    monkeypatch.setattr(
        mb.urllib.request, "urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(_http_error(403, body)),
    )
    with pytest.raises(mb.FaucetAlreadyFunded) as excinfo:
        mb._post_faucet("http://f", dest_hex="0x00", amount=1)
    assert excinfo.value.free_balance == 42


def test_post_faucet_connection_error_is_transient(monkeypatch):
    monkeypatch.setattr(
        mb.urllib.request, "urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(urllib.error.URLError("refused")),
    )
    with pytest.raises(mb.FaucetTransientError):
        mb._post_faucet("http://f", dest_hex="0x00", amount=1)


# ---------------------------------------------------------------------------
# Idempotent registration (_ensure_registered)
# ---------------------------------------------------------------------------


class _RegClient:
    """Stub client for _ensure_registered: scripted query_miner + submit."""

    def __init__(self, miner_info, receipt_error=None):
        self._miner_info = miner_info
        self._receipt_error = receipt_error
        self.submits = []

    async def query_miner(self, _account):
        return self._miner_info

    async def submit_extrinsic(self, **kwargs):
        self.submits.append(kwargs)
        return SimpleNamespace(error=self._receipt_error)


async def test_ensure_registered_already_registered_is_noop():
    info = SimpleNamespace(deposit=1, proofs_submitted=2, proofs_won=3)
    client = _RegClient(info)
    newly = await mb._ensure_registered(client, _stub_keystore())
    assert newly is False
    assert client.submits == [], "must not re-submit when already registered"


async def test_ensure_registered_submits_when_absent():
    client = _RegClient(None)
    newly = await mb._ensure_registered(client, _stub_keystore())
    assert newly is True
    assert len(client.submits) == 1
    assert client.submits[0]["call_function"] == "register_miner"


async def test_ensure_registered_raises_on_receipt_error():
    client = _RegClient(None, receipt_error="DispatchError")
    with pytest.raises(RuntimeError, match="register_miner failed"):
        await mb._ensure_registered(client, _stub_keystore())
