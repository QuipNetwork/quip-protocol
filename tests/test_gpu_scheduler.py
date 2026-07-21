# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gate 3 tests: GPU scheduler SM budget and yielding behavior."""

import os

import pytest

cp = pytest.importorskip(
    "cupy", reason="CuPy required for GPU scheduler tests",
)


class TestSmBudget:
    """Verify SM budget management and yielding modes."""

    def test_not_yielding_static_budget(self):
        """yielding=False returns static budget from config."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=50,
            yielding=False,
        )
        assert sched.get_sm_budget() == 40
        sched.stop()

    def test_not_yielding_no_throttle(self):
        """yielding=False never throttles."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=100,
            yielding=False,
        )
        assert not sched.should_throttle()
        sched.stop()

    def test_yielding_without_nvml_uses_static(self):
        """yielding=True without pynvml falls back to static."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=75,
            yielding=True,
        )
        # Without a working NVML handle, falls back to static
        budget = sched.get_sm_budget()
        assert budget == 60  # 80 * 75 / 100
        sched.stop()

    def test_yielding_static_budget_ignores_load(self):
        """SM budget is always static — NVML load is ignored.

        NVML reports total GPU utilization including our own
        kernels, so adaptive budgeting causes self-throttle.
        Budget is always the static ceiling from config.
        """
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=100,
            gpu_utilization_pct=80,
            yielding=True,
        )
        # Even with simulated high external load,
        # budget stays at static ceiling
        sched._nvml_handle = "fake"
        sched._util_value.value = 50

        budget = sched.get_sm_budget()
        assert budget == 80
        sched.stop()

    def test_yielding_high_load_throttles(self):
        """High external load triggers throttle."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=100,
            gpu_utilization_pct=80,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        sched._util_value.value = 95
        assert sched.should_throttle()
        sched.stop()

    def test_yielding_low_load_no_throttle(self):
        """Low external load does not throttle."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=100,
            gpu_utilization_pct=80,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        sched._util_value.value = 30
        assert not sched.should_throttle()
        sched.stop()

    def test_budget_always_at_least_one(self):
        """Even with 100% external load, budget >= 1."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=100,
            gpu_utilization_pct=80,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        sched._util_value.value = 100
        assert sched.get_sm_budget() >= 1
        sched.stop()

    def test_yielding_property(self):
        from GPU.gpu_scheduler import KernelScheduler

        s_on = KernelScheduler(yielding=True)
        s_off = KernelScheduler(yielding=False)
        assert s_on.yielding is True
        assert s_off.yielding is False
        s_on.stop()
        s_off.stop()

    def test_default_yielding_is_false(self):
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler()
        assert sched.yielding is False
        sched.stop()

    def test_stop_is_idempotent(self):
        """Calling stop() twice doesn't raise."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(yielding=False)
        sched.stop()
        sched.stop()  # should not raise


class TestBuildGpuMinerCfg:
    """Verify per-device config merging logic."""

    def test_empty_section_returns_empty(self):
        from shared.miner_core import _build_gpu_miner_cfg
        assert _build_gpu_miner_cfg({}) == {}

    def test_extracts_known_keys(self):
        from shared.miner_core import _build_gpu_miner_cfg

        section = {
            "utilization": 80,
            "yielding": True,
            "backend": "local",  # not a miner key
        }
        result = _build_gpu_miner_cfg(section)
        assert result == {
            "utilization": 80,
            "yielding": True,
        }

    def test_per_device_overrides_common(self):
        from shared.miner_core import _build_gpu_miner_cfg

        common = _build_gpu_miner_cfg({
            "utilization": 80,
            "yielding": False,
        })
        dev_cfg = _build_gpu_miner_cfg(
            {"utilization": 50, "yielding": True},
            defaults=common,
        )
        assert dev_cfg == {
            "utilization": 50,
            "yielding": True,
        }

    def test_partial_override_keeps_defaults(self):
        from shared.miner_core import _build_gpu_miner_cfg

        common = _build_gpu_miner_cfg({
            "utilization": 80,
            "yielding": True,
        })
        dev_cfg = _build_gpu_miner_cfg(
            {"utilization": 50},
            defaults=common,
        )
        assert dev_cfg == {
            "utilization": 50,
            "yielding": True,
        }

    def test_defaults_used_when_no_override(self):
        from shared.miner_core import _build_gpu_miner_cfg

        common = _build_gpu_miner_cfg({
            "utilization": 100,
        })
        dev_cfg = _build_gpu_miner_cfg(
            {},
            defaults=common,
        )
        assert dev_cfg == {"utilization": 100}


class TestMpsConfiguration:
    """Verify MPS env var is set based on gpu_utilization."""

    def test_mps_env_set_without_yielding(self, monkeypatch):
        """CUDA_MPS_ACTIVE_THREAD_PERCENTAGE set when
        gpu_utilization < 100, yielding=False.
        """
        from GPU.gpu_scheduler import configure_mps_thread_limit

        monkeypatch.delenv(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            raising=False,
        )
        configure_mps_thread_limit(
            gpu_utilization_pct=50,
            device_id=0,
            yielding=False,
        )
        assert (
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]
            == "50"
        )

    def test_mps_env_not_set_at_100_pct(self, monkeypatch):
        """Env var NOT set when gpu_utilization=100,
        yielding=False.
        """
        from GPU.gpu_scheduler import configure_mps_thread_limit

        monkeypatch.delenv(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            raising=False,
        )
        configure_mps_thread_limit(
            gpu_utilization_pct=100,
            device_id=0,
            yielding=False,
        )
        assert (
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            not in os.environ
        )


class TestSmsPerNonceReachesGibbs:
    """Verify sms_per_nonce config reaches the sampler."""

    def test_sms_per_nonce_reaches_gibbs(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(
            sms_per_nonce=8,
        )
        assert sampler._sf_sms_per_nonce_val == 8

    def test_gpu_util_scales_sm_ceiling(self):
        """CudaMiner with 50% util gets half the SMs."""
        dev_id = 0
        device_sms = cp.cuda.Device(
            dev_id,
        ).attributes['MultiProcessorCount']

        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sm_ceiling = max(1, int(device_sms * 50 / 100))
        sampler = CudaGibbsSampler(
            max_sms=sm_ceiling,
        )
        assert sampler.max_sms == sm_ceiling
        assert sm_ceiling <= device_sms // 2 + 1
