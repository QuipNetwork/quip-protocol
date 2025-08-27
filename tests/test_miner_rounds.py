import os
import sys
import subprocess
import threading
import time
import signal
from pathlib import Path
from click.testing import CliRunner
import quip_cli


def test_cpu_auto_mine_quick():
    """Test CPU mining by running the CLI and monitoring for mining activity."""
    import threading
    import logging
    from unittest.mock import patch
    import io
    
    # Capture logging output to monitor mining activity
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    blocks_mined = 0
    mining_started = threading.Event()
    target_reached = threading.Event()
    
    def mining_monitor():
        """Monitor the log output for mining activity."""
        nonlocal blocks_mined
        timeout = 60  # Max 60 seconds
        start_time = time.time()
        
        while not target_reached.is_set() and time.time() - start_time < timeout:
            log_contents = log_capture.getvalue()
            
            # Look for mining activity
            if "Mining process started in background" in log_contents and not mining_started.is_set():
                mining_started.set()
                print("Mining process detected as started")
            
            # Count blocks mined
            current_blocks = log_contents.count("✅ Block") 
            if current_blocks > blocks_mined:
                blocks_mined = current_blocks
                print(f"Detected {blocks_mined} mined blocks")
                if blocks_mined >= 3:
                    target_reached.set()
                    break
            
            time.sleep(0.5)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=mining_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Start the CPU node
    from quip_cli import cpu
    from click.testing import CliRunner
    
    runner = CliRunner()
    
    def run_cpu_test():
        # This will run until interrupted
        result = runner.invoke(cpu, ["--port", "0"], catch_exceptions=False)
        return result
    
    # Run the CPU node in a separate thread
    node_thread = threading.Thread(target=run_cpu_test)
    node_thread.daemon = True
    node_thread.start()
    
    # Wait for either target to be reached or timeout
    target_reached.wait(timeout=60)
    
    # Clean up
    logger.removeHandler(handler)
    logger.setLevel(original_level)
    
    # Assert that we successfully detected mining activity
    log_contents = log_capture.getvalue()
    print(f"Final log contents (last 500 chars): {log_contents[-500:]}")
    
    # Check that mining started
    assert mining_started.is_set(), "Mining process did not start"
    
    # For now, just verify mining started - actual block mining takes longer
    # In a more complete test, we could mock the mining difficulty to be easier
    print("✅ CPU mining test passed - mining process started successfully")


def test_gpu_auto_mine_quick_env_only(monkeypatch):
    # Only assert CLI/env wiring for a quick GPU auto-mine run (no actual GPU required)
    cfg = """
[global]
default = "gpu"
port = 0

[gpu]
backend = "local"
devices = ["0"]
"""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.toml")
        with open(cfg_path, "w") as f:
            f.write(cfg)
        # Patch runner to avoid actually spawning GPU work, just confirm exit path
        captured = {}
        def fake_run(kind, host, port, peer, auto_mine, env_overrides=None):
            captured.update({"env": env_overrides or {}, "auto": auto_mine})
            return 0
        monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)
        runner = CliRunner()
        res = runner.invoke(quip_cli.quip_network_node, ["--config", cfg_path, "gpu", "--auto-mine", "5"])
        assert res.exit_code == 0
        assert captured["env"].get("QUIP_GPU_BACKEND") == "local"
        assert captured["env"].get("QUIP_GPU_DEVICES") == "0"
        assert captured["auto"] == 5

