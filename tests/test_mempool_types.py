"""Unit tests for `substrate.mempool_types` — pure dataclass round-trips.

Live-chain tests for solver registration / job proposal live in
`tests/test_mempool_client.py`; this file only exercises the encoders +
decoders so it runs without docker.
"""
from __future__ import annotations

import pytest

from substrate.mempool_types import (
    IsingParams,
    JobMode,
    MinerType,
    OrderStatus,
    ResultDelivery,
    RewardResolution,
    solutions_to_scale,
)


# ----------------------------------------------------------------------
# MinerType
# ----------------------------------------------------------------------


def test_miner_type_from_kind_cpu_gpu_qpu():
    assert MinerType.from_kind("cpu") == MinerType.CPU
    assert MinerType.from_kind("CPU") == MinerType.CPU
    assert MinerType.from_kind("gpu") == MinerType.GPU
    # bare 'qpu' maps to QpuDwave so the existing CPU/GPU/QPU triplet
    # works without callers learning the vendor variants.
    assert MinerType.from_kind("qpu") == MinerType.QPU_DWAVE
    assert MinerType.from_kind("dwave") == MinerType.QPU_DWAVE


def test_miner_type_from_kind_unknown_raises():
    with pytest.raises(ValueError, match="unknown miner kind"):
        MinerType.from_kind("fpga")


def test_miner_type_scale_variant_roundtrip():
    for mt in MinerType:
        assert MinerType.from_scale_variant(mt.to_scale_variant()) == mt


def test_miner_type_unknown_scale_variant_raises():
    with pytest.raises(ValueError, match="unknown SCALE MinerType"):
        MinerType.from_scale_variant("QpuFake")


# ----------------------------------------------------------------------
# Tagged-enum variants
# ----------------------------------------------------------------------


def test_reward_resolution_single_best_scale_dict():
    r = RewardResolution.single_best()
    assert r.to_scale_dict() == {"SingleBest": None}


def test_reward_resolution_top_n_requires_n():
    with pytest.raises(ValueError, match="requires `n`"):
        RewardResolution(tag="TopNWeighted").to_scale_dict()


def test_reward_resolution_from_string_value():
    """The substrate-interface decoder may surface bare-variant enums as a
    plain string when there are no inner fields. Accept both shapes."""
    r = RewardResolution.from_scale_value("SingleBest")
    assert r.tag == "SingleBest"
    assert r.n is None


def test_reward_resolution_from_dict_value():
    r = RewardResolution.from_scale_value({"TopNEqual": {"n": 5}})
    assert r.tag == "TopNEqual"
    assert r.n == 5


def test_result_delivery_on_chain_only():
    assert ResultDelivery.on_chain_only().to_scale_dict() == {"OnChainOnly": None}


def test_result_delivery_callback_requires_endpoint():
    with pytest.raises(ValueError, match="requires `endpoint`"):
        ResultDelivery(tag="Callback").to_scale_dict()


def test_job_mode_open():
    assert JobMode.open().to_scale_dict() == {"Open": None}


def test_job_mode_bid_requires_at_least_one_field():
    """Pallet rejects empty Bid via `EmptyBidCriteria` — fail at construction."""
    with pytest.raises(ValueError, match="at least one"):
        JobMode.bid()


def test_job_mode_bid_with_accounts_only():
    miners = (b"\x11" * 32, b"\x22" * 32)
    m = JobMode.bid(miners=miners)
    d = m.to_scale_dict()
    assert d["Bid"]["miners"] == ["0x" + m.hex() for m in miners]
    assert d["Bid"]["miner_types"] is None


def test_job_mode_bid_with_types_only():
    m = JobMode.bid(miner_types=(MinerType.CPU, MinerType.GPU))
    d = m.to_scale_dict()
    assert d["Bid"]["miners"] is None
    assert d["Bid"]["miner_types"] == ["Cpu", "Gpu"]


def test_job_mode_bid_with_both():
    miner_bytes = b"\x33" * 32
    m = JobMode.bid(miners=(miner_bytes,), miner_types=(MinerType.QPU_DWAVE,))
    d = m.to_scale_dict()
    assert d["Bid"]["miners"] == ["0x" + miner_bytes.hex()]
    assert d["Bid"]["miner_types"] == ["QpuDwave"]


# ----------------------------------------------------------------------
# IsingParams
# ----------------------------------------------------------------------


def _toy_ising() -> IsingParams:
    return IsingParams(
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2)),
        h_values=(1000, -500, 0),     # millivalues: 1.0, -0.5, 0.0
        j_values=(-250, 750),
        min_energy_milli=-3000,
        min_diversity_milli=200,
        min_solutions=5,
    )


def test_ising_params_length_validation():
    """The pallet implicitly assumes h.len == nodes.len and j.len == edges.len —
    fail fast on the Python side so a bad call doesn't make it to the chain."""
    with pytest.raises(ValueError, match="h_values length"):
        IsingParams(
            nodes=(0, 1),
            edges=((0, 1),),
            h_values=(1,),         # mismatched: 1 != 2 nodes
            j_values=(1,),
        )
    with pytest.raises(ValueError, match="j_values length"):
        IsingParams(
            nodes=(0, 1),
            edges=((0, 1),),
            h_values=(1, 2),
            j_values=(1, 2),       # mismatched: 2 != 1 edge
        )


def test_ising_params_scale_dict_shape():
    """The BoundedVec composites need 1-element tuple wrapping for
    substrate-interface to encode them correctly (same rule Phase 4 hit
    with register_topology)."""
    p = _toy_ising()
    d = p.to_scale_dict()
    assert d["nodes"] == ([0, 1, 2],)
    assert d["edges"] == ([(0, 1), (1, 2)],)
    assert d["h_values"] == ([1000, -500, 0],)
    assert d["j_values"] == ([-250, 750],)
    assert d["min_energy_milli"] == -3000
    assert d["min_diversity_milli"] == 200
    assert d["min_solutions"] == 5


def test_ising_params_decode_roundtrip():
    """Decode from a substrate-interface-shaped dict (note: storage
    decoding returns the inner Vecs, NOT the 1-tuple-wrapped composites)."""
    p = _toy_ising()
    storage_value = {
        "nodes": list(p.nodes),
        "edges": [list(e) for e in p.edges],
        "h_values": list(p.h_values),
        "j_values": list(p.j_values),
        "min_energy_milli": p.min_energy_milli,
        "min_diversity_milli": p.min_diversity_milli,
        "min_solutions": p.min_solutions,
    }
    decoded = IsingParams.from_scale_value(storage_value)
    assert decoded == p


def test_ising_params_decode_with_missing_optionals():
    """All three quality-floor fields are Option<T> on the chain and may
    be absent from the decoded value."""
    storage_value = {
        "nodes": [0, 1],
        "edges": [[0, 1]],
        "h_values": [0, 0],
        "j_values": [0],
        "min_energy_milli": None,
        "min_diversity_milli": None,
        "min_solutions": None,
    }
    decoded = IsingParams.from_scale_value(storage_value)
    assert decoded.min_energy_milli is None
    assert decoded.min_diversity_milli is None
    assert decoded.min_solutions is None


# ----------------------------------------------------------------------
# MempoolJobContext
# ----------------------------------------------------------------------


def test_mempool_job_context_from_job_order():
    """from_job_order must copy all IsingParams fields into the context."""
    from substrate.mempool_types import (
        IsingParams, JobMode, JobOrder, MempoolJobContext,
        OrderStatus, OrderTiming, ResultDelivery, RewardResolution,
    )
    ising = IsingParams(
        nodes=(10, 20),
        edges=((10, 20),),
        h_values=(500, -500),
        j_values=(250,),
        min_energy_milli=-1000,
        min_diversity_milli=100,
        min_solutions=2,
    )
    order = JobOrder(
        spec_id=b"\x01" * 32,
        proposer=b"\x02" * 32,
        ising_params=ising,
        reward=1000,
        mode=JobMode.open(),
        resolution=RewardResolution.single_best(),
        timing=OrderTiming(deadline_blocks=10, block_wait=1),
        delivery=ResultDelivery.on_chain_only(),
        status=OrderStatus.OPENED,
        created_at=100,
        first_solution_at=None,
        solution_count=0,
    )
    ctx = MempoolJobContext.from_job_order(order_id=5, order=order)
    assert ctx.order_id == 5
    assert ctx.nodes == (10, 20)
    assert ctx.edges == ((10, 20),)
    assert ctx.h_values == (500, -500)
    assert ctx.j_values == (250,)
    assert ctx.min_energy_milli == -1000
    assert ctx.min_diversity_milli == 100
    assert ctx.min_solutions == 2


def test_mempool_job_context_rejects_mismatched_h_values():
    """MempoolJobContext.__post_init__ must raise if h_values length != nodes."""
    from substrate.mempool_types import MempoolJobContext
    with pytest.raises(ValueError, match="h_values length"):
        MempoolJobContext(
            order_id=1,
            nodes=(0, 1, 2),
            edges=(),
            h_values=(0, 0),       # 2 != 3 nodes
            j_values=(),
        )


def test_mempool_job_context_rejects_mismatched_j_values():
    """MempoolJobContext.__post_init__ must raise if j_values length != edges."""
    from substrate.mempool_types import MempoolJobContext
    with pytest.raises(ValueError, match="j_values length"):
        MempoolJobContext(
            order_id=1,
            nodes=(0, 1),
            edges=((0, 1),),
            h_values=(0, 0),
            j_values=(0, 0),       # 2 != 1 edge
        )


# ----------------------------------------------------------------------
# solutions_to_scale
# ----------------------------------------------------------------------


def test_solutions_to_scale_validates_spin_values():
    assert solutions_to_scale([[1, -1, 0, 1]]) == [[1, -1, 0, 1]]
    with pytest.raises(ValueError, match=r"solutions\[0\]\[2\] = 2"):
        solutions_to_scale([[1, -1, 2]])
    with pytest.raises(ValueError, match=r"solutions\[1\]\[0\] = -2"):
        solutions_to_scale([[1, -1], [-2, 1]])


def test_solutions_to_scale_copies_input():
    """The helper returns a fresh list-of-lists so callers can safely
    mutate the returned structure without affecting their input."""
    inp = [[1, -1, 0]]
    out = solutions_to_scale(inp)
    out[0][0] = 99
    assert inp[0][0] == 1


# ----------------------------------------------------------------------
# OrderStatus
# ----------------------------------------------------------------------


def test_order_status_from_scale_variant_roundtrip():
    assert OrderStatus.from_scale_variant("Opened") == OrderStatus.OPENED
    assert OrderStatus.from_scale_variant("Expired") == OrderStatus.EXPIRED
    assert OrderStatus.from_scale_variant("Closed") == OrderStatus.CLOSED


def test_order_status_from_scale_variant_unknown_raises_value_error():
    with pytest.raises(ValueError, match="unknown SCALE OrderStatus variant"):
        OrderStatus.from_scale_variant("Disputed")


# ----------------------------------------------------------------------
# RewardResolution encode paths
# ----------------------------------------------------------------------


def test_reward_resolution_top_n_weighted_scale_dict():
    r = RewardResolution.top_n_weighted(3)
    assert r.to_scale_dict() == {"TopNWeighted": {"n": 3}}


def test_reward_resolution_top_n_equal_scale_dict():
    r = RewardResolution.top_n_equal(5)
    assert r.to_scale_dict() == {"TopNEqual": {"n": 5}}


def test_reward_resolution_from_string_unknown_raises():
    with pytest.raises(ValueError, match="bare-string RewardResolution"):
        RewardResolution.from_scale_value("TopNWeighted")


# ----------------------------------------------------------------------
# ResultDelivery encode paths
# ----------------------------------------------------------------------


def test_result_delivery_callback_scale_dict():
    r = ResultDelivery.callback(b"http://example.com")
    assert r.to_scale_dict() == {"Callback": {"endpoint": b"http://example.com"}}


def test_result_delivery_callback_with_poll_scale_dict():
    r = ResultDelivery.callback_with_poll(b"http://example.com")
    assert r.to_scale_dict() == {"CallbackWithPoll": {"endpoint": b"http://example.com"}}
