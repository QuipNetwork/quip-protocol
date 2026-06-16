from __future__ import annotations

from shared.system_info import build_descriptor
from substrate.miner_registry import descriptor_call_params, participation_call_params


def test_descriptor_call_params_encodes_bounded_runtime_shape():
    descriptor = build_descriptor(
        node_id="rig",
        node_name="rig",
        public_host="rig.example.com",
        public_port=20050,
        rpc_endpoints=["https://rig.example.com/rpc"],
        auto_mine=True,
        log_level="INFO",
        miner_specs=[
            {"id": "rig-CPU-1", "kind": "cpu", "args": {}},
            {"id": "rig-GPU-1", "kind": "cuda", "args": {"device": "0"}},
            {
                "id": "rig-QPU-1",
                "kind": "qpu",
                "cfg": {"qpu_type": "ibm", "daily_budget": "60s"},
            },
        ],
        include_system_info=False,
    )

    params = descriptor_call_params(descriptor, node_id="rig")
    body = params["descriptor"]["V1"]

    assert body["node_id"] == b"rig"
    assert body["node_name"] == b"rig"
    assert body["public_host"] == b"rig.example.com"
    assert body["public_port"] == 20050
    assert body["rpc_endpoints"] == [b"https://rig.example.com/rpc"]
    assert body["auto_mine"] is True
    assert body["log_level"] == "Info"
    assert [m["kind"] for m in body["miners"]] == ["Cpu", "Gpu", "QpuIbm"]
    assert body["miners"][0]["label"] == b"cpu"


def test_participation_call_params_targets_candidate_qblock():
    assert participation_call_params(
        latest_qblock_id=None,
        kind="cpu",
        budget_seconds=None,
    ) == {
        "qblock_id": 1,
        "kind": "Cpu",
        "budget_seconds": None,
    }

    assert participation_call_params(
        latest_qblock_id=41,
        kind="qpu",
        budget_seconds=90.7,
    ) == {
        "qblock_id": 42,
        "kind": "QpuDwave",
        "budget_seconds": 90,
    }
