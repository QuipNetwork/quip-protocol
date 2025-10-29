#!/usr/bin/env python3
"""Test script for signal-responsive mining architecture."""

import multiprocessing as mp
import pytest
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.miner_worker import MinerHandle

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
        import GPU.metal_sampler
        test_sampler = GPU.metal_sampler.MetalSampler("mps")
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
        mock_block = type('MockBlock', (), {
            'hash': b'test_hash',
            'header': type('Header', (), {'index': 0})()
        })()
        
        mock_node_info = type('NodeInfo', (), {
            'miner_id': 'test-miner'
        })()
        
        mock_requirements = type('Requirements', (), {
            'difficulty_energy': -14000.0,
            'min_diversity': 0.3,
            'min_solutions': 10
        })()
        
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
            
            # Let it mine for 1 second, then signal stop
            time.sleep(1.0)
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
            
            # Let it mine for 1 second, then signal stop
            time.sleep(1.0)
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
            
            # Let it mine for 1 second, then signal stop
            time.sleep(1.0)
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
            
            # Let it mine for 1 second, then signal stop
            time.sleep(1.0)
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
            
            # Let it mine for 1 second, then signal stop
            time.sleep(1.0)
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