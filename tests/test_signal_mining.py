#!/usr/bin/env python3
"""Test script for signal-responsive mining architecture."""

import multiprocessing as mp
import pytest
import time
import sys
import os
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.miner_worker import MinerHandle


# Module-level mock classes that can be pickled
@dataclass
class MockHeader:
    index: int = 0

@dataclass
class MockBlock:
    hash: bytes = b'test_hash'
    header: MockHeader = None

    def __post_init__(self):
        if self.header is None:
            self.header = MockHeader()

@dataclass
class MockNodeInfo:
    miner_id: str = 'test-miner'

@dataclass
class MockRequirements:
    difficulty_energy: float = -14000.0
    min_diversity: float = 0.2
    min_solutions: int = 10

def _cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import GPU.sampler
        test_sampler = GPU.sampler.GPUSampler("0")
        del test_sampler
        return True
    except Exception:
        return False

def _qpu_available():
    """Check if QPU is available for testing."""
    try:
        import QPU.dwave_sampler
        test_sampler = QPU.dwave_sampler.DWaveSamplerWrapper()
        del test_sampler
        return True
    except Exception:
        return False

def _metal_available():
    """Check if Metal is available for testing."""
    try:
        import platform
        if platform.system() != "Darwin":
            return False
        from GPU.metal_sa import MetalSASampler
        test_sampler = MetalSASampler()
        del test_sampler
        return True
    except Exception:
        return False

def _modal_available():
    """Check if Modal is available for testing."""
    try:
        import GPU.modal_sampler
        # Modal requires cloud access, skip for now
        return False
    except Exception:
        return False

class TestSignalResponsiveMining:
    """Test suite for signal-responsive mining architecture."""

    @pytest.fixture
    def mock_mining_objects(self):
        """Create mock objects for testing."""
        mock_block = MockBlock()
        mock_node_info = MockNodeInfo()
        mock_requirements = MockRequirements()
        return mock_block, mock_node_info, mock_requirements
    
    def test_cpu_signal_responsiveness(self, mock_mining_objects):
        """Test that CPU miner responds to signals within 200ms."""
        mock_block, mock_node_info, mock_requirements = mock_mining_objects
        
        spec = {
            "id": "test-cpu-signal",
            "kind": "cpu",
            "args": {},
            "cfg": {}
        }
        
        miner_handle = MinerHandle(spec)
        
        try:
            stop_event = mp.Event()
            
            # Let the worker spin up briefly, then signal stop
            time.sleep(0.2)
            signal_time = time.time()
            stop_event.set()
            
            # Test signal-responsive mining
            result = miner_handle.mine_with_timeout(
                mock_block, mock_node_info, mock_requirements, 
                int(time.time()), stop_event
            )
            
            end_time = time.time()
            response_time = (end_time - signal_time) * 1000
            
            assert response_time <= 200, f"CPU signal response time {response_time:.1f}ms exceeds 200ms limit"
            assert result is None, "Should return None when stopped"
            
        finally:
            miner_handle.close()
    
    @pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
    def test_cuda_signal_responsiveness(self, mock_mining_objects):
        """Test that CUDA miner responds to signals within 200ms."""
        mock_block, mock_node_info, mock_requirements = mock_mining_objects
        
        spec = {
            "id": "test-cuda-signal",
            "kind": "cuda",
            "args": {"device": "0"},
            "cfg": {}
        }
        
        miner_handle = MinerHandle(spec)
        
        try:
            stop_event = mp.Event()
            
            # Let the worker spin up briefly, then signal stop
            time.sleep(0.2)
            signal_time = time.time()
            stop_event.set()
            
            # Test signal-responsive mining
            result = miner_handle.mine_with_timeout(
                mock_block, mock_node_info, mock_requirements,
                int(time.time()), stop_event
            )
            
            end_time = time.time()
            response_time = (end_time - signal_time) * 1000
            
            assert response_time <= 200, f"CUDA signal response time {response_time:.1f}ms exceeds 200ms limit"
            assert result is None, "Should return None when stopped"
            
        finally:
            miner_handle.close()
    
    @pytest.mark.skipif(not _qpu_available(), reason="QPU not available")
    def test_qpu_signal_responsiveness(self, mock_mining_objects):
        """Test that QPU miner responds to signals within 200ms."""
        mock_block, mock_node_info, mock_requirements = mock_mining_objects
        
        spec = {
            "id": "test-qpu-signal",
            "kind": "qpu",
            "args": {},
            "cfg": {}
        }
        
        miner_handle = MinerHandle(spec)
        
        try:
            stop_event = mp.Event()
            
            # Let the worker spin up briefly, then signal stop
            time.sleep(0.2)
            signal_time = time.time()
            stop_event.set()
            
            # Test signal-responsive mining
            result = miner_handle.mine_with_timeout(
                mock_block, mock_node_info, mock_requirements,
                int(time.time()), stop_event
            )
            
            end_time = time.time()
            response_time = (end_time - signal_time) * 1000
            
            assert response_time <= 200, f"QPU signal response time {response_time:.1f}ms exceeds 200ms limit"
            assert result is None, "Should return None when stopped"
            
        finally:
            miner_handle.close()
    
    @pytest.mark.skipif(not _metal_available(), reason="Metal not available")
    def test_metal_signal_responsiveness(self, mock_mining_objects):
        """Test that Metal miner responds to signals within 200ms."""
        mock_block, mock_node_info, mock_requirements = mock_mining_objects
        
        spec = {
            "id": "test-metal-signal",
            "kind": "metal",
            "args": {},
            "cfg": {}
        }
        
        miner_handle = MinerHandle(spec)
        
        try:
            stop_event = mp.Event()
            
            # Let the worker spin up briefly, then signal stop
            time.sleep(0.2)
            signal_time = time.time()
            stop_event.set()
            
            # Test signal-responsive mining
            result = miner_handle.mine_with_timeout(
                mock_block, mock_node_info, mock_requirements,
                int(time.time()), stop_event
            )
            
            end_time = time.time()
            response_time = (end_time - signal_time) * 1000
            
            assert response_time <= 200, f"Metal signal response time {response_time:.1f}ms exceeds 200ms limit"
            assert result is None, "Should return None when stopped"
            
        finally:
            miner_handle.close()
    
    @pytest.mark.skipif(not _modal_available(), reason="Modal not available")
    def test_modal_signal_responsiveness(self, mock_mining_objects):
        """Test that Modal miner responds to signals within 200ms."""
        mock_block, mock_node_info, mock_requirements = mock_mining_objects
        
        spec = {
            "id": "test-modal-signal",
            "kind": "modal",
            "args": {"gpu_type": "t4"},
            "cfg": {}
        }
        
        miner_handle = MinerHandle(spec)
        
        try:
            stop_event = mp.Event()
            
            # Let the worker spin up briefly, then signal stop
            time.sleep(0.2)
            signal_time = time.time()
            stop_event.set()
            
            # Test signal-responsive mining
            result = miner_handle.mine_with_timeout(
                mock_block, mock_node_info, mock_requirements,
                int(time.time()), stop_event
            )
            
            end_time = time.time()
            response_time = (end_time - signal_time) * 1000
            
            assert response_time <= 200, f"Modal signal response time {response_time:.1f}ms exceeds 200ms limit"
            assert result is None, "Should return None when stopped"
            
        finally:
            miner_handle.close()


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"])