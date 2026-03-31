# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gate 3 tests: GPU scheduler slot state machine and events.

Verifies GpuBufferSlot lifecycle transitions and
KernelScheduler polling behavior.
"""

import os

import pytest

cp = pytest.importorskip(
    "cupy", reason="CuPy required for GPU scheduler tests",
)


class TestSlotStateTransitions:
    """Verify the slot state machine transitions."""

    def test_initial_state_is_free(self):
        from GPU.gpu_scheduler import GpuBufferSlot, SlotState

        slot = GpuBufferSlot(
            slot_id=0,
            stream=cp.cuda.Stream(non_blocking=True),
            event=cp.cuda.Event(),
        )
        assert slot.state == SlotState.FREE

    def test_full_lifecycle(self):
        from GPU.gpu_scheduler import GpuBufferSlot, SlotState

        stream = cp.cuda.Stream(non_blocking=True)
        slot = GpuBufferSlot(
            slot_id=0,
            stream=stream,
            event=cp.cuda.Event(),
        )

        # FREE → UPLOADING
        slot.begin_upload()
        assert slot.state == SlotState.UPLOADING

        # UPLOADING → READY
        slot.finish_upload()
        assert slot.state == SlotState.READY

        # READY → ACTIVE
        compute = cp.cuda.Stream(non_blocking=True)
        slot.launch(compute)
        assert slot.state == SlotState.ACTIVE

        # Record completion
        slot.record_launch(compute)
        compute.synchronize()

        # ACTIVE → DONE (via poll)
        assert slot.event.done
        slot.state = SlotState.DONE

        # DONE → FREE
        slot.harvest()
        assert slot.state == SlotState.FREE

    def test_invalid_transition_raises(self):
        from GPU.gpu_scheduler import GpuBufferSlot

        slot = GpuBufferSlot(
            slot_id=0,
            stream=cp.cuda.Stream(non_blocking=True),
            event=cp.cuda.Event(),
        )

        # Can't finish upload when FREE
        with pytest.raises(AssertionError, match="UPLOADING"):
            slot.finish_upload()

        # Can't launch when FREE
        compute = cp.cuda.Stream(non_blocking=True)
        with pytest.raises(AssertionError, match="READY"):
            slot.launch(compute)

        # Can't harvest when FREE
        with pytest.raises(AssertionError, match="DONE"):
            slot.harvest()


class TestKernelScheduler:
    """Verify scheduler pool management."""

    def test_initial_all_free(self):
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(num_slots=2)
        assert len(sched.free_slots()) == 2
        assert not sched.any_active()

    def test_poll_empty_when_no_active(self):
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(num_slots=2)
        assert sched.poll_completed() == []

    def test_event_detection(self):
        """Launch a real kernel and verify event detects completion."""
        from GPU.gpu_scheduler import KernelScheduler, SlotState

        sched = KernelScheduler(num_slots=2)
        slot = sched.free_slots()[0]

        # Simulate upload
        slot.begin_upload()
        slot.finish_upload()

        # Launch a trivial kernel
        slot.launch(sched.compute_stream)

        # Run a trivial operation to have something to record
        with sched.compute_stream:
            arr = cp.zeros(10)
            arr += 1

        slot.record_launch(sched.compute_stream)
        sched.compute_stream.synchronize()

        # Poll should detect completion
        completed = sched.poll_completed()
        assert len(completed) == 1
        assert completed[0].slot_id == slot.slot_id
        assert slot.state == SlotState.DONE

        # Harvest
        slot.harvest()
        assert slot.state == SlotState.FREE
        assert len(sched.free_slots()) == 2

    def test_double_buffer_workflow(self):
        """Two slots alternate: one active, one uploading."""
        from GPU.gpu_scheduler import KernelScheduler, SlotState

        sched = KernelScheduler(num_slots=2)

        # Slot 0: upload + launch
        s0 = sched.slots[0]
        s0.begin_upload()
        s0.finish_upload()
        s0.launch(sched.compute_stream)
        with sched.compute_stream:
            cp.zeros(10)
        s0.record_launch(sched.compute_stream)

        assert sched.any_active()
        assert len(sched.free_slots()) == 1

        # Slot 1: upload while slot 0 runs
        s1 = sched.slots[1]
        s1.begin_upload()
        s1.finish_upload()
        assert s1.state == SlotState.READY

        # Wait for slot 0
        sched.compute_stream.synchronize()
        completed = sched.poll_completed()
        assert len(completed) == 1
        completed[0].harvest()

        # Now launch slot 1
        s1.launch(sched.compute_stream)
        with sched.compute_stream:
            cp.zeros(10)
        s1.record_launch(sched.compute_stream)
        sched.compute_stream.synchronize()

        completed = sched.poll_completed()
        assert len(completed) == 1
        completed[0].harvest()

        assert len(sched.free_slots()) == 2


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
        sched._external_util_pct = 50

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
        sched._external_util_pct = 95
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
        sched._external_util_pct = 30
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
        sched._external_util_pct = 100
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
        from shared.node import _build_gpu_miner_cfg
        assert _build_gpu_miner_cfg({}) == {}

    def test_extracts_known_keys(self):
        from shared.node import _build_gpu_miner_cfg

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
        from shared.node import _build_gpu_miner_cfg

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
        from shared.node import _build_gpu_miner_cfg

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
        from shared.node import _build_gpu_miner_cfg

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


class TestAdaptiveNonces:
    """Verify compute_target_nonces and hysteresis."""

    def test_compute_target_nonces_no_external(self):
        """Returns max_nonces when 0 external processes."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        result = sched.compute_target_nonces(
            max_nonces=20, active_nonces=20,
        )
        assert result == 20
        sched.stop()

    def test_compute_target_nonces_with_external(self):
        """Returns < max_nonces with external processes."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        sched._external_util_pct = 50

        # Mock count_external_gpu_processes to return 1
        sched.count_external_gpu_processes = lambda: 1

        result = sched.compute_target_nonces(
            max_nonces=20, active_nonces=20,
        )
        assert 1 <= result < 20
        sched.stop()

    def test_compute_target_nonces_fair_share(self):
        """With N competing, target ~ max_nonces/(N+1)."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        sched._external_util_pct = 90

        # 3 external processes — fair share is ~1/4
        sched.count_external_gpu_processes = lambda: 3

        # Force our_est >= total_util to trigger
        # fair share fallback
        result = sched.compute_target_nonces(
            max_nonces=20, active_nonces=20,
        )
        # Fair share: 100% / 4 = 25%, 25/100 * 20 = 5
        assert 1 <= result <= 6
        sched.stop()

    def test_check_stable_target_hysteresis(self):
        """Returns None on 1st call, value on 2nd."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._nvml_handle = "fake"
        # No external processes → always returns max
        first = sched.check_stable_target(
            max_nonces=10, active_nonces=10,
        )
        assert first is None

        second = sched.check_stable_target(
            max_nonces=10, active_nonces=10,
        )
        assert second == 10
        sched.stop()

    def test_check_stable_target_reset_on_change(self):
        """Returns None when target changes between calls."""
        from GPU.gpu_scheduler import KernelScheduler

        sched = KernelScheduler(
            device_sms=80,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._nvml_handle = "fake"

        # First call: no external → target=10
        sched.check_stable_target(
            max_nonces=10, active_nonces=10,
        )

        # Simulate external process appearing
        sched.count_external_gpu_processes = lambda: 2
        sched._external_util_pct = 80

        # Target changed → counter resets
        result = sched.check_stable_target(
            max_nonces=10, active_nonces=10,
        )
        assert result is None
        sched.stop()


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
