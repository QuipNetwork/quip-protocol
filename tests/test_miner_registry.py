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

    params = descriptor_call_params(descriptor, node_id="rig", schema_version=1)
    body = params["descriptor"]["V1"]

    # Every BoundedVec field is wrapped in a 1-tuple: the runtime metadata
    # describes BoundedVec as a composite with a single inner Vec field and
    # scalecodec (1.2.x) does not flatten it. Passing the bare bytes/list
    # fails to encode with "Element count of value (0) doesn't match
    # type_definition (1)", so the wrapping is load-bearing — assert it
    # explicitly rather than the flat shape that never encoded.
    assert body["node_id"] == (b"rig",)
    assert body["node_name"] == (b"rig",)
    assert body["public_host"] == (b"rig.example.com",)
    assert body["public_port"] == 20050
    assert body["rpc_endpoints"] == ([b"https://rig.example.com/rpc"],)
    assert body["auto_mine"] is True
    assert body["log_level"] == "Info"

    # miners is itself a BoundedVec, so the spec list is wrapped too.
    (miners,) = body["miners"]
    assert [m["kind"] for m in miners] == ["Cpu", "Gpu", "QpuIbm"]
    assert miners[0]["label"] == (b"cpu",)
    assert miners[0]["device_id"] == (b"rig-CPU-1",)


def test_descriptor_call_params_v2_carries_system_info_and_runtime():
    """V2 (the default) records system_info + runtime; BoundedVec<u8> fields
    inside the nested structs are 1-tuple-wrapped, nested structs are plain
    dicts, and gpus (a BoundedVec) is wrapped."""
    from shared.system_info import CPUInfo, GPUInfo, NodeDescriptor, Runtime, SystemInfo

    descriptor = NodeDescriptor(
        descriptor_version=1,
        node_name="rig",
        public_host=None,
        public_port=None,
        rpc_endpoints=[],
        auto_mine=True,
        log_level="INFO",
        runtime=Runtime(
            python="3.13.1", quip_version="0.2.1", protocol_version=2,
            in_docker=True, docker_image="quip-node:rc",
        ),
        miners={"cpu": {"kind": "cpu"}},
        system_info=SystemInfo(
            os={"system": "Linux", "release": "6.1.0", "machine": "x86_64"},
            cpu=CPUInfo(logical_cores=16, physical_cores=8,
                        brand="AMD EPYC 7763", arch="x86_64"),
            memory_mb=64000,
            gpus=[GPUInfo(index=0, vendor="NVIDIA", name="H100",
                          memory_mb=81920, observed_utilization_pct=150)],
        ),
    )

    body = descriptor_call_params(descriptor, node_id="rig")["descriptor"]["V2"]
    si = body["system_info"]
    assert si["os"]["system"] == (b"Linux",)
    assert si["cpu"]["brand"] == (b"AMD EPYC 7763",)
    assert si["cpu"]["logical_cores"] == 16            # Option<u32>, not wrapped
    assert si["memory_mb"] == 64000
    (gpus,) = si["gpus"]                                # gpus is a BoundedVec
    assert gpus[0]["vendor"] == (b"NVIDIA",)
    assert gpus[0]["utilization_pct"] == 100           # clamped from 150

    rt = body["runtime"]
    assert rt["python"] == (b"3.13.1",)
    assert rt["protocol_version"] == 2
    assert rt["in_docker"] is True
    assert rt["docker_image"] == (b"quip-node:rc",)


def test_descriptor_call_params_v2_system_info_none_when_absent():
    """--no-system-info / no survey → system_info is None (Option::None)."""
    from shared.system_info import NodeDescriptor, Runtime

    descriptor = NodeDescriptor(
        descriptor_version=1, node_name="rig", public_host=None,
        public_port=None, rpc_endpoints=[], auto_mine=True, log_level="INFO",
        runtime=Runtime(python="3.13.1", quip_version="0.2.1",
                        protocol_version=2, in_docker=False, docker_image=None),
        miners={"cpu": {"kind": "cpu"}}, system_info=None,
    )
    body = descriptor_call_params(descriptor, node_id="rig")["descriptor"]["V2"]
    assert body["system_info"] is None
    assert body["runtime"]["docker_image"] is None


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
