"""Click-based CLI for quip-protocol.

Provides two console commands:
- quip-network-node: run a single P2P node (cpu/gpu/qpu) backed by shared.network_node.NetworkNode
- quip-network-simulator: launch multiple nodes using quip-network-node and connect them locally to each other
"""
from __future__ import annotations

import os
import multiprocessing
import signal
import subprocess
import sys
import json
import time
import asyncio
from typing import Any, Dict, Optional, List

import click
import traceback


# TOML loader supporting Python 3.10 via tomli and 3.11+ via tomllib
try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore

from shared.node import Node
from shared.network_node import NetworkNode
from shared.block import load_genesis_block
from shared.version import get_version
from shared.logging_config import setup_logging


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    with open(path, "rb") as f:
        config = _toml.load(f)

    # Set DWave environment variables from TOML config if present
    qpu_config = config.get("qpu", {})
    if "dwave_api_key" in qpu_config:
        os.environ["DWAVE_API_KEY"] = qpu_config["dwave_api_key"]
        os.environ["DWAVE_API_TOKEN"] = qpu_config["dwave_api_key"]

    if "dwave_api_solver" in qpu_config:
        os.environ["DWAVE_API_SOLVER"] = qpu_config["dwave_api_solver"]

    if "dwave_region_url" in qpu_config:
        os.environ["DWAVE_REGION_URL"] = qpu_config["dwave_region_url"]

    cfg = _merge_globals_from_toml(config)
    if qpu_config:
        cfg["qpu"] = qpu_config
    if "gpu" in config:
        cfg["gpu"] = config["gpu"]
    if "cpu" in config:
        cfg["cpu"] = config["cpu"]

    _print_final_config(cfg, "load_config")

    return cfg


def _merge_globals_from_toml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten [global] section of TOML into NetworkNode config keys.
    Leaves 'cpu', 'gpu', 'qpu' sections as-is.
    """
    if not cfg:
        return {}
    g = dict(cfg.get("global", {}) or {})
    out: Dict[str, Any] = {}
    for k, v in g.items():
        if k not in ["cpu", "gpu", "qpu"]:
            out[k] = v
        else:
            out[f"global.k"] = v
    return out


def _print_final_config(config: Dict[str, Any], miner_type: str):
    """Print the final configuration as JSON for debugging."""
    # Create a clean copy for display
    display_config = dict(config)

    # Add metadata
    display_config["_miner_type"] = miner_type
    display_config["_config_source"] = "merged_toml_and_cli"

    click.echo("Final configuration:")
    click.echo(json.dumps(display_config, indent=2, default=str))
    click.echo()



def _apply_global_overrides(conf: Dict[str, Any],
                             listen: Optional[str],
                             port: Optional[int],
                             public_host: Optional[str],
                             node_name: Optional[str],
                             secret: Optional[str],
                             auto_mine: Optional[bool],
                             peers: Optional[List[str]],
                             timeout: Optional[int],
                             heartbeat_interval: Optional[int],
                             heartbeat_timeout: Optional[int],
                             fanout: Optional[int],
                             log_level: Optional[str] = None,
                             node_log: Optional[str] = None,
                             http_log: Optional[str] = None) -> Dict[str, Any]:
    c = dict(conf)
    if listen is not None:
        c["listen"] = listen
    if port is not None:
        c["port"] = int(port)
    if public_host is not None:
        c["public_host"] = public_host
    if node_name is not None:
        c["node_name"] = node_name
    if secret is not None:
        c["secret"] = secret
    if auto_mine is not None:
        c["auto_mine"] = bool(auto_mine)
    if peers:
        c["peer"] = list(peers)
    if timeout is not None:
        c["node_timeout"] = int(timeout)
    if heartbeat_interval is not None:
        c["heartbeat_interval"] = int(heartbeat_interval)
    if heartbeat_timeout is not None:
        c["heartbeat_timeout"] = int(heartbeat_timeout)
    if fanout is not None:
        c["fanout"] = int(fanout)
    if log_level is not None:
        c["log_level"] = log_level
    if node_log is not None:
        c["node_log"] = node_log
    if http_log is not None:
        c["http_log"] = http_log
    return c


async def _async_run_network_node(config: Dict[str, Any], genesis_config_file: str) -> int:
    """Create NetworkNode with genesis, start server/tasks, and run until Ctrl-C."""
    # Setup logging before creating NetworkNode
    log_level = config.get("log_level", "INFO")
    node_log_file = config.get("node_log")
    http_log_file = config.get("http_log")
    node_name = config.get("node_name", "quip-node")

    # Setup logging with our custom configuration
    setup_logging(
        log_level=log_level,
        node_log_file=node_log_file,
        http_log_file=http_log_file,
        node_name=node_name
    )

    # Load genesis and pass to NetworkNode constructor
    genesis = load_genesis_block(genesis_config_file)
    node = NetworkNode(config, genesis)

    # Note: NetworkNode creates its own logger in constructor with proper node ID
    # The setup_logging loggers are kept for other components that may need them

    await node.start()
    try:
        # Run until interrupted
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        click.echo("Interrupted by user")
    finally:
        await node.stop()
    return 0


def _run_network_node_sync(config: Dict[str, Any], genesis_config_file: str) -> int:
    try:
        return asyncio.run(_async_run_network_node(config, genesis_config_file))
    except KeyboardInterrupt:
        click.echo("Interrupted by user")
        return 130
    except Exception as e:
        click.echo(f"Error: {e}")
        click.echo(traceback.format_exc())
        return 1


# -----------------------------
# quip-network-node
# -----------------------------

@click.group(invoke_without_command=True)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), help="Path to TOML config file")
@click.option("--version", is_flag=True, help="Show version and exit")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def quip_network_node(ctx: click.Context, config: Optional[str], version: bool, debug_config: bool):
    """Run a single quip network node.

    Subcommands: cpu, gpu, qpu

    If invoked without a subcommand, --config may specify [global].default
    to choose a default subcommand. Global settings provide listen/port/peer/auto_mine.
    """
    if version:
        click.echo(f"quip-protocol {get_version()}")
        return

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["config"] = _load_config(config)
    ctx.obj["debug_config"] = debug_config

    if ctx.invoked_subcommand is None:
        cfg = ctx.obj.get("config", {})

        # Check if any miner sections are present
        has_miners = any(k in cfg for k in ("cpu", "gpu", "qpu"))
        if not has_miners:
            raise click.UsageError("No subcommand given and no miner sections ([cpu], [gpu], [qpu]) found in config")
        
        # Apply debug config from global options
        if ctx.obj.get("debug_config", False):
            _print_final_config(cfg, "auto-configured")
        
        # Use genesis_block.json as default genesis config
        genesis_config = cfg.get("genesis_config", "genesis_block.json")
        
        sys.exit(_run_network_node_sync(cfg, genesis_config))


# Subcommands: cpu/gpu/qpu. Each builds a NetworkNode config from TOML and CLI flags.

@quip_network_node.command(name="cpu")
# Global network options
@click.option("--listen", type=str, default=None, help="Address to bind (defaults from [global].listen or 127.0.0.1)")
@click.option("--port", type=int, default=None, help="Port to bind (defaults from [global].port or 20049)")
@click.option("--public-host", type=str, default=None, help="Public host:port advertised to peers")
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option("--secret", type=str, default=None, help="Deterministic secret for keypair")
@click.option("--auto-mine/--no-auto-mine", default=None, help="Enable/disable auto-mining when no peers found")
@click.option("--peer", "peers", multiple=True, help="Peer host:port (repeat for multiple)")
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option("--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats")
@click.option("--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds")
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Logging level")
@click.option("--node-log", type=str, default=None, help="Path to main node log file (defaults to stderr)")
@click.option("--http-log", type=str, default=None, help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)")
# CPU options
@click.option("--num-cpus", type=int, default=None, help="Number of CPU miners to spawn (default 1)")
# Other
@click.option("--genesis-config", type=str, default="genesis_block.json", show_default=True, help="Genesis block configuration file")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def cpu(
    ctx: click.Context,
    listen: Optional[str],
    port: Optional[int],
    public_host: Optional[str],
    node_name: Optional[str],
    secret: Optional[str],
    auto_mine: Optional[bool],
    peers: List[str],
    timeout: Optional[int],
    heartbeat_interval: Optional[int],
    heartbeat_timeout: Optional[int],
    fanout: Optional[int],
    log_level: Optional[str],
    node_log: Optional[str],
    http_log: Optional[str],
    num_cpus: Optional[int],
    genesis_config: str,
    debug_config: bool,
):
    # Load full TOML config
    toml_cfg = (ctx.obj or {}).get("config", {})

    # Filter to CPU-only by removing other miner sections
    conf = dict(toml_cfg)
    conf.pop("gpu", None)
    conf.pop("qpu", None)

    conf = _apply_global_overrides(conf, listen, port, public_host, node_name, secret, auto_mine, list(peers) or None, timeout, heartbeat_interval, heartbeat_timeout, fanout, log_level, node_log, http_log)

    # Handle CPU-specific configuration
    cpu_cfg = dict((conf.get("cpu") or {}))
    if num_cpus is not None:
        cpu_cfg["num_cpus"] = int(num_cpus)
    if not cpu_cfg:
        cpu_cfg = {"num_cpus": 1}
    conf["cpu"] = cpu_cfg


    # Use genesis config from TOML if CLI option is default and TOML has it
    if genesis_config == "genesis_block.json" and "genesis_config" in conf:
        genesis_config = conf["genesis_config"]

    # Print final configuration if requested
    if debug_config or (ctx.obj or {}).get("debug_config", False):
        _print_final_config(conf, "cpu")

    sys.exit(_run_network_node_sync(conf, genesis_config))


@quip_network_node.command(name="gpu")
# Global network options
@click.option("--listen", type=str, default=None, help="Address to bind (defaults from [global].listen or 127.0.0.1)")
@click.option("--port", type=int, default=None, help="Port to bind (defaults from [global].port or 20049)")
@click.option("--public-host", type=str, default=None, help="Public host:port advertised to peers")
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option("--secret", type=str, default=None, help="Deterministic secret for keypair")
@click.option("--auto-mine/--no-auto-mine", default=None, help="Enable/disable auto-mining when no peers found")
@click.option("--peer", "peers", multiple=True, help="Peer host:port (repeat for multiple)")
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option("--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats")
@click.option("--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds")
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Logging level")
@click.option("--node-log", type=str, default=None, help="Path to main node log file (defaults to stderr)")
@click.option("--http-log", type=str, default=None, help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)")
# GPU options
@click.option("--gpu-backend", type=click.Choice(["local", "modal", "mps"], case_sensitive=False), default=None, help="GPU backend: local|modal|mps")
@click.option("--device", "devices", multiple=True, help="GPU device(s) for local backend (e.g., 0 1)")
@click.option("--gpu-type", "gpu_types", multiple=True, help="GPU type(s) for modal backend (e.g., t4 a10g)")
@click.option("--gpu-utilization", type=int, default=100, help="GPU utilization percentage (1-100, default: 100)")
# Other
@click.option("--genesis-config", type=str, default="genesis_block.json", show_default=True, help="Genesis block configuration file")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def gpu(
    ctx: click.Context,
    listen: Optional[str],
    port: Optional[int],
    public_host: Optional[str],
    node_name: Optional[str],
    secret: Optional[str],
    auto_mine: Optional[bool],
    peers: List[str],
    timeout: Optional[int],
    heartbeat_interval: Optional[int],
    heartbeat_timeout: Optional[int],
    fanout: Optional[int],
    log_level: Optional[str],
    node_log: Optional[str],
    http_log: Optional[str],
    gpu_backend: Optional[str],
    devices: List[str],
    gpu_types: List[str],
    gpu_utilization: int,
    genesis_config: str,
    debug_config: bool,
):
    """Run a GPU-only network node."""
    # Load full TOML config
    toml_cfg = (ctx.obj or {}).get("config", {})

    # Filter to GPU-only by removing other miner sections
    conf = dict(toml_cfg)
    conf.pop("cpu", None)
    conf.pop("qpu", None)

    # Apply CLI overrides
    conf = _apply_global_overrides(conf, listen, port, public_host, node_name, secret, auto_mine, list(peers) or None, timeout, heartbeat_interval, heartbeat_timeout, fanout, log_level, node_log, http_log)

    # Handle GPU-specific configuration
    gpu_cfg = dict((conf.get("gpu") or {}))
    if gpu_backend is not None:
        gpu_cfg["backend"] = str(gpu_backend).lower()
    if devices:
        gpu_cfg["devices"] = [str(d) for d in devices]
    if gpu_types:
        gpu_cfg["types"] = [str(t) for t in gpu_types]
    if gpu_utilization != 100:
        gpu_cfg["gpu_utilization"] = gpu_utilization
    if not gpu_cfg:
        gpu_cfg = {"backend": "local"}

    # Default to device 0 if backend is local and no devices specified
    backend = gpu_cfg.get("backend", "local")
    if backend == "local" and "devices" not in gpu_cfg:
        gpu_cfg["devices"] = ["0"]

    conf["gpu"] = gpu_cfg

    # Use genesis config from TOML if CLI option is default and TOML has it
    if genesis_config == "genesis_block.json" and "genesis_config" in conf:
        genesis_config = conf["genesis_config"]

    # Print final configuration if requested
    if debug_config or (ctx.obj or {}).get("debug_config", False):
        _print_final_config(conf, "gpu")

    sys.exit(_run_network_node_sync(conf, genesis_config))


@quip_network_node.command(name="qpu")
# Global network options
@click.option("--listen", type=str, default=None, help="Address to bind (defaults from [global].listen or 127.0.0.1)")
@click.option("--port", type=int, default=None, help="Port to bind (defaults from [global].port or 20049)")
@click.option("--public-host", type=str, default=None, help="Public host:port advertised to peers")
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option("--secret", type=str, default=None, help="Deterministic secret for keypair")
@click.option("--auto-mine/--no-auto-mine", default=None, help="Enable/disable auto-mining when no peers found")
@click.option("--peer", "peers", multiple=True, help="Peer host:port (repeat for multiple)")
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option("--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats")
@click.option("--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds")
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Logging level")
@click.option("--node-log", type=str, default=None, help="Path to main node log file (defaults to stderr)")
@click.option("--http-log", type=str, default=None, help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)")
# QPU options
@click.option("--dwave-api-key", type=str, default=None, help="D-Wave API key")
@click.option("--dwave-api-solver", type=str, default=None, help="D-Wave solver name")
@click.option("--dwave-region-url", type=str, default=None, help="D-Wave SAPI region endpoint URL")
# Other
@click.option("--genesis-config", type=str, default="genesis_block.json", show_default=True, help="Genesis block configuration file")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def qpu(
    ctx: click.Context,
    listen: Optional[str],
    port: Optional[int],
    public_host: Optional[str],
    node_name: Optional[str],
    secret: Optional[str],
    auto_mine: Optional[bool],
    peers: List[str],
    timeout: Optional[int],
    heartbeat_interval: Optional[int],
    heartbeat_timeout: Optional[int],
    fanout: Optional[int],
    log_level: Optional[str],
    node_log: Optional[str],
    http_log: Optional[str],
    dwave_api_key: Optional[str],
    dwave_api_solver: Optional[str],
    dwave_region_url: Optional[str],
    genesis_config: str,
    debug_config: bool,
):
    """Run a QPU-only network node."""
    # Load full TOML config
    toml_cfg = (ctx.obj or {}).get("config", {})

    conf = dict(toml_cfg)
    conf.pop("gpu", None)
    conf.pop("cpu", None)

    # Apply CLI overrides
    conf = _apply_global_overrides(conf, listen, port, public_host, node_name, secret, auto_mine, list(peers) or None, timeout, heartbeat_interval, heartbeat_timeout, fanout, log_level, node_log, http_log)

    # Handle QPU-specific configuration
    qpu_cfg = dict((conf.get("qpu") or {}))

    # Set environment variables from CLI arguments
    if dwave_api_key is not None:
        qpu_cfg["dwave_api_key"] = dwave_api_key
        # Set environment variable for child processes
        os.environ["DWAVE_API_KEY"] = dwave_api_key
        os.environ["DWAVE_API_TOKEN"] = dwave_api_key
    if dwave_api_solver is not None:
        qpu_cfg["dwave_api_solver"] = dwave_api_solver
        # Set environment variable for child processes
        os.environ["DWAVE_API_SOLVER"] = dwave_api_solver
    if dwave_region_url is not None:
        qpu_cfg["dwave_region_url"] = dwave_region_url
        # Set environment variable for child processes
        os.environ["DWAVE_REGION_URL"] = dwave_region_url

    # Ensure environment variables are set from TOML config values (if not already set by CLI)
    if "dwave_api_key" in qpu_cfg and "DWAVE_API_KEY" not in os.environ:
        os.environ["DWAVE_API_KEY"] = qpu_cfg["dwave_api_key"]
        os.environ["DWAVE_API_TOKEN"] = qpu_cfg["dwave_api_key"]
    if "dwave_api_solver" in qpu_cfg and "DWAVE_API_SOLVER" not in os.environ:
        os.environ["DWAVE_API_SOLVER"] = qpu_cfg["dwave_api_solver"]
    if "dwave_region_url" in qpu_cfg and "DWAVE_REGION_URL" not in os.environ:
        os.environ["DWAVE_REGION_URL"] = qpu_cfg["dwave_region_url"]

    if not qpu_cfg:
        qpu_cfg = {}
    conf["qpu"] = qpu_cfg

    # Use genesis config from TOML if CLI option is default and TOML has it
    if genesis_config == "genesis_block.json" and "genesis_config" in conf:
        genesis_config = conf["genesis_config"]

    # Print final configuration if requested
    if debug_config or (ctx.obj or {}).get("debug_config", False):
        _print_final_config(conf, "qpu")

    sys.exit(_run_network_node_sync(conf, genesis_config))


# -----------------------------
# quip-network-simulator
# -----------------------------

@click.group(name="quip-network-simulator", invoke_without_command=True)
@click.option("--scenario", type=click.Choice(["mixed", "cpu", "gpu"], case_sensitive=False), default="mixed", show_default=True, help="Network scenario to launch")
@click.option("--num-cpu", type=int, default=None, help="Override: number of CPU nodes")
@click.option("--num-gpu", type=int, default=None, help="Override: number of GPU nodes")
@click.option("--num-qpu", type=int, default=None, help="Override: number of QPU nodes")
@click.option("--base-port", type=int, default=8080, show_default=True, help="Starting port for first node")
@click.option("--print-only", is_flag=True, help="Only print commands, do not execute")
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def quip_network_simulator(ctx: click.Context, scenario: str, num_cpu: Optional[int], num_gpu: Optional[int], num_qpu: Optional[int], base_port: int, print_only: bool, version: bool):
    """Launch a local multi-node network using quip-network-node (separate processes).

    Subcommands:
      smoketest [cpu|gpu-local|gpu-metal|gpu-modal|qpu]  Run a single-node smoke test
    """
    if version:
        click.echo(f"quip-protocol {get_version()}")
        return

    if ctx.invoked_subcommand is not None:
        return

    scenario = scenario.lower()
    # Defaults modeled after launch_network.py
    if scenario == "mixed":
        cpu_n = 3 if num_cpu is None else num_cpu
        gpu_n = 2 if num_gpu is None else num_gpu
        qpu_n = 1 if num_qpu is None else num_qpu
    elif scenario == "cpu":
        cpu_n = 4 if num_cpu is None else num_cpu
        gpu_n = 0 if num_gpu is None else num_gpu
        qpu_n = 0 if num_qpu is None else num_qpu
    elif scenario == "gpu":
        cpu_n = 0 if num_cpu is None else num_cpu
        gpu_n = 4 if num_gpu is None else num_gpu
        qpu_n = 0 if num_qpu is None else num_qpu
    else:
        raise click.ClickException(f"Unknown scenario: {scenario}")

    cmds = []
    port = base_port
    # Bootstrap preference: if any CPU nodes, make first CPU bootstrap, else GPU, else QPU
    order = [("cpu", cpu_n), ("gpu", gpu_n), ("qpu", qpu_n)]
    # Determine which kind will be used for bootstrap
    bootstrap_kind = next((k for k, n in order if n > 0), None)
    if bootstrap_kind is None:
        raise click.ClickException("Nothing to launch (all counts are zero)")

    processes = []

    def _cmd_for(kind: str, port: int, peer: Optional[str]) -> list[str]:
        base = ["quip-network-node", kind, "--port", str(port)]
        if peer:
            base += ["--peer", peer]
        return base

    # Build command list
    peer_addr = None
    for kind, count in order:
        for _ in range(count):
            if kind == bootstrap_kind and peer_addr is None:
                cmds.append(_cmd_for(kind, port, None))
                peer_addr = f"localhost:{port}"
            else:
                cmds.append(_cmd_for(kind, port, peer_addr))
            port += 1

    # Print commands
    for c in cmds:
        click.echo("Running: " + " ".join(c))

    if print_only:
        return

    # Spawn all processes; terminate on Ctrl+C
    try:
        for c in cmds:
            p = subprocess.Popen(c)
            processes.append(p)
        click.echo("\nNetwork is running. Press Ctrl+C to stop all nodes.")
        # Wait indefinitely
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


@quip_network_simulator.command(name="smoketest")
@click.argument("target", type=click.Choice(["cpu", "gpu-local", "gpu-metal", "gpu-modal", "qpu"], case_sensitive=False))
@click.option("--print-only", is_flag=True, help="Only print command, do not execute")
def quip_network_smoketest(target: str, print_only: bool):
    """Run a single-node smoke test.

    Targets:
      cpu, gpu-local, gpu-metal, gpu-modal, qpu
    """
    target = target.lower()
    if target == "cpu":
        cmd = ["python", "-m", "tests.smoke_node_cpu_only"]
    elif target == "gpu-local":
        cmd = ["python", "-m", "tests.smoke_node_gpu_local"]
    elif target == "gpu-metal":
        cmd = ["python", "-m", "tests.smoke_node_gpu_metal"]
    elif target == "gpu-modal":
        cmd = ["python", "-m", "tests.smoke_node_gpu_modal"]
    elif target == "qpu":
        cmd = ["python", "-m", "tests.smoke_node_qpu"]
    else:
        raise click.ClickException(f"Unknown smoketest: {target}")
    click.echo("Running: " + " ".join(cmd))
    if print_only:
        return
    try:
        p = subprocess.Popen(cmd)
        p.wait()
    except KeyboardInterrupt:
        pass


# Entry points for console_scripts

def network_node_main():
    # Set multiprocessing start method to 'spawn' to avoid context mixing issues
    multiprocessing.set_start_method('spawn', force=True)

    quip_network_node(standalone_mode=False)

# -----------------------------
# quip-node-stats (experimental)
# -----------------------------

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), help="Path to TOML config file")
@click.option("--interval", type=float, default=5.0, help="Seconds between stats prints")
@click.pass_context
def quip_node_stats(_: click.Context, config: Optional[str], interval: float):
    """Run a single in-process node and periodically print stats to stdout.

    This is an experimental helper that constructs Node directly from TOML config
    and prints Node.get_stats() every --interval seconds. Ctrl-C to stop.
    """
    # Set multiprocessing start method to 'spawn' to avoid context mixing issues
    multiprocessing.set_start_method('spawn', force=True)

    cfg = _load_config(config)
    miners_config = cfg or {}
    genesis_block=load_genesis_block(cfg.get("genesis_config", "genesis_block.json"))
    node = Node(node_id="stats-node", miners_config=miners_config, genesis_block=genesis_block)
    click.echo("Starting stats loop (Ctrl-C to stop)...")
    try:
        while True:
            stats = node.get_stats()
            click.echo(json.dumps(stats, indent=2))
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("Stopping...")
    finally:
        node.close()


def network_simulator_main():
    # Set multiprocessing start method to 'spawn' to avoid context mixing issues
    multiprocessing.set_start_method('spawn', force=True)

    quip_network_simulator(standalone_mode=False)

