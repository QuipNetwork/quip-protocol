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


def _set_dwave_env(section: Dict[str, Any]) -> None:
    """Set D-Wave env vars from a config section."""
    token = section.get("token")
    if token and "DWAVE_API_KEY" not in os.environ:
        os.environ["DWAVE_API_KEY"] = token
        os.environ["DWAVE_API_TOKEN"] = token
    solver = section.get("solver")
    if solver and "DWAVE_API_SOLVER" not in os.environ:
        os.environ["DWAVE_API_SOLVER"] = solver
    region = section.get("dwave_region_url")
    if region and "DWAVE_REGION_URL" not in os.environ:
        os.environ["DWAVE_REGION_URL"] = region


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    with open(path, "rb") as f:
        config = _toml.load(f)

    # Set D-Wave environment variables from [dwave] section if present
    dwave_config = config.get("dwave", {})
    if isinstance(dwave_config, dict):
        _set_dwave_env(dwave_config)

    cfg = _merge_globals_from_toml(config)

    # Forward miner sections
    for section in ("cpu", "gpu", "qpu"):
        if section in config:
            cfg[section] = config[section]

    # Forward device-type sections (top-level [cuda.N], [metal], [dwave], etc.)
    _device_sections = (
        "cuda", "nvidia", "metal", "modal",
        "dwave", "ibm", "braket", "pasqal", "ionq", "origin",
    )
    for section in _device_sections:
        if section in config:
            cfg[section] = config[section]

    # Forward telemetry API configuration
    if "telemetry_api" in config:
        cfg["telemetry_api"] = config["telemetry_api"]

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
                             public_port: Optional[int],
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
                             http_log: Optional[str] = None,
                             rest_port: Optional[int] = None,
                             rest_insecure_port: Optional[int] = None) -> Dict[str, Any]:
    c = dict(conf)
    if listen is not None:
        c["listen"] = listen
    if port is not None:
        c["port"] = int(port)
    if public_host is not None:
        c["public_host"] = public_host
    if public_port is not None:
        c["public_port"] = int(public_port)
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
    if rest_port is not None:
        c["rest_port"] = int(rest_port)
    if rest_insecure_port is not None:
        c["rest_insecure_port"] = int(rest_insecure_port)
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
    # Install uvloop for 2-4x async throughput if available
    from shared.event_loop import install_uvloop_policy
    install_uvloop_policy()

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
@click.option("--listen", type=str, default=None, help="Address to bind; IPv6 supported (e.g., ::1 or ::). Defaults from [global].listen or 127.0.0.1")
@click.option("--port", type=int, default=None, help="Port to bind (defaults from [global].port or 20049)")
@click.option("--public-host", type=str, default=None, help="Public hostname or IP advertised to peers")
@click.option("--public-port", type=int, default=None, help="Public port advertised to peers (defaults to --port)")
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option("--secret", type=str, default=None, help="Deterministic secret for keypair")
@click.option("--auto-mine/--no-auto-mine", default=None, help="Enable/disable auto-mining when no peers found")
@click.option("--peer", "peers", multiple=True, help="Peer host:port; use [IPv6]:port for IPv6 (repeat for multiple)")
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option("--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats")
@click.option("--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds")
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Logging level")
@click.option("--node-log", type=str, default=None, help="Path to main node log file (defaults to stderr)")
@click.option("--http-log", type=str, default=None, help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)")
# REST API options
@click.option("--rest-port", type=int, default=None, help="REST API HTTPS port (-1 disables, defaults from [global].rest_port or -1)")
@click.option("--rest-insecure-port", type=int, default=None, help="REST API HTTP port (-1 disables, defaults from [global].rest_insecure_port or 20050)")
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
    public_port: Optional[int],
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
    rest_port: Optional[int],
    rest_insecure_port: Optional[int],
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

    conf = _apply_global_overrides(conf, listen, port, public_host, public_port, node_name, secret, auto_mine, list(peers) or None, timeout, heartbeat_interval, heartbeat_timeout, fanout, log_level, node_log, http_log, rest_port, rest_insecure_port)

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
@click.option("--listen", type=str, default=None, help="Address to bind; IPv6 supported (e.g., ::1 or ::). Defaults from [global].listen or 127.0.0.1")
@click.option("--port", type=int, default=None, help="Port to bind (defaults from [global].port or 20049)")
@click.option("--public-host", type=str, default=None, help="Public hostname or IP advertised to peers")
@click.option("--public-port", type=int, default=None, help="Public port advertised to peers (defaults to --port)")
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option("--secret", type=str, default=None, help="Deterministic secret for keypair")
@click.option("--auto-mine/--no-auto-mine", default=None, help="Enable/disable auto-mining when no peers found")
@click.option("--peer", "peers", multiple=True, help="Peer host:port; use [IPv6]:port for IPv6 (repeat for multiple)")
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option("--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats")
@click.option("--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds")
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Logging level")
@click.option("--node-log", type=str, default=None, help="Path to main node log file (defaults to stderr)")
@click.option("--http-log", type=str, default=None, help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)")
# REST API options
@click.option("--rest-port", type=int, default=None, help="REST API HTTPS port (-1 disables, defaults from [global].rest_port or -1)")
@click.option("--rest-insecure-port", type=int, default=None, help="REST API HTTP port (-1 disables, defaults from [global].rest_insecure_port or 20050)")
# GPU options
@click.option("--gpu-backend", type=click.Choice(["local", "modal", "mps"], case_sensitive=False), default=None, help="GPU backend: local|modal|mps")
@click.option("--device", "devices", multiple=True, help="GPU device(s) for local backend (e.g., 0 1)")
@click.option("--gpu-type", "gpu_types", multiple=True, help="GPU type(s) for modal backend (e.g., t4 a10g)")
@click.option("--gpu-utilization", type=int, default=100, help="GPU utilization percentage (1-100, default: 100)")
@click.option("--yielding", is_flag=True, default=False, help="Yield GPU to other processes (NVML-adaptive nonce scaling)")
# Other
@click.option("--genesis-config", type=str, default="genesis_block.json", show_default=True, help="Genesis block configuration file")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def gpu(
    ctx: click.Context,
    listen: Optional[str],
    port: Optional[int],
    public_host: Optional[str],
    public_port: Optional[int],
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
    rest_port: Optional[int],
    rest_insecure_port: Optional[int],
    gpu_backend: Optional[str],
    devices: List[str],
    gpu_types: List[str],
    gpu_utilization: int,
    yielding: bool,
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
    conf = _apply_global_overrides(conf, listen, port, public_host, public_port, node_name, secret, auto_mine, list(peers) or None, timeout, heartbeat_interval, heartbeat_timeout, fanout, log_level, node_log, http_log, rest_port, rest_insecure_port)

    # Build GPU config from CLI args as top-level device sections.
    # [gpu] holds global defaults; [cuda.N]/[metal]/[modal] hold devices.
    gpu_cfg = dict((conf.get("gpu") or {}))
    if gpu_utilization != 100:
        gpu_cfg["utilization"] = gpu_utilization
    if yielding:
        gpu_cfg["yielding"] = True
    conf["gpu"] = gpu_cfg

    backend = str(gpu_backend or "cuda").lower()
    if backend in ("local", "cuda", "nvidia"):
        dev_list = [str(d) for d in devices] if devices else ["0"]
        cuda_section = conf.get("cuda") or {}
        for d in dev_list:
            if d not in cuda_section:
                cuda_section[d] = {}
        conf["cuda"] = cuda_section
    elif backend == "mps":
        if "metal" not in conf:
            conf["metal"] = {}
    elif backend == "modal":
        modal_cfg = conf.get("modal") or {}
        if gpu_types:
            modal_cfg["gpu_type"] = str(gpu_types[0])
        conf["modal"] = modal_cfg

    # Use genesis config from TOML if CLI option is default and TOML has it
    if genesis_config == "genesis_block.json" and "genesis_config" in conf:
        genesis_config = conf["genesis_config"]

    # Print final configuration if requested
    if debug_config or (ctx.obj or {}).get("debug_config", False):
        _print_final_config(conf, "gpu")

    sys.exit(_run_network_node_sync(conf, genesis_config))


@quip_network_node.command(name="qpu")
# Global network options
@click.option("--listen", type=str, default=None, help="Address to bind; IPv6 supported (e.g., ::1 or ::). Defaults from [global].listen or 127.0.0.1")
@click.option("--port", type=int, default=None, help="Port to bind (defaults from [global].port or 20049)")
@click.option("--public-host", type=str, default=None, help="Public hostname or IP advertised to peers")
@click.option("--public-port", type=int, default=None, help="Public port advertised to peers (defaults to --port)")
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option("--secret", type=str, default=None, help="Deterministic secret for keypair")
@click.option("--auto-mine/--no-auto-mine", default=None, help="Enable/disable auto-mining when no peers found")
@click.option("--peer", "peers", multiple=True, help="Peer host:port; use [IPv6]:port for IPv6 (repeat for multiple)")
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option("--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats")
@click.option("--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds")
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Logging level")
@click.option("--node-log", type=str, default=None, help="Path to main node log file (defaults to stderr)")
@click.option("--http-log", type=str, default=None, help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)")
# REST API options
@click.option("--rest-port", type=int, default=None, help="REST API HTTPS port (-1 disables, defaults from [global].rest_port or -1)")
@click.option("--rest-insecure-port", type=int, default=None, help="REST API HTTP port (-1 disables, defaults from [global].rest_insecure_port or 20050)")
# QPU options
@click.option("--dwave-api-key", type=str, default=None, help="D-Wave API key")
@click.option("--dwave-api-solver", type=str, default=None, help="D-Wave solver name")
@click.option("--dwave-region-url", type=str, default=None, help="D-Wave SAPI region endpoint URL")
# QPU time budget options
@click.option("--qpu-daily-budget", type=str, default=None, help="Daily QPU time budget (e.g., 40s, 2m) - calculate from your Leap allocation")
# Other
@click.option("--genesis-config", type=str, default="genesis_block.json", show_default=True, help="Genesis block configuration file")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def qpu(
    ctx: click.Context,
    listen: Optional[str],
    port: Optional[int],
    public_host: Optional[str],
    public_port: Optional[int],
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
    rest_port: Optional[int],
    rest_insecure_port: Optional[int],
    dwave_api_key: Optional[str],
    dwave_api_solver: Optional[str],
    dwave_region_url: Optional[str],
    qpu_daily_budget: Optional[str],
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
    conf = _apply_global_overrides(conf, listen, port, public_host, public_port, node_name, secret, auto_mine, list(peers) or None, timeout, heartbeat_interval, heartbeat_timeout, fanout, log_level, node_log, http_log, rest_port, rest_insecure_port)

    # Build QPU config — CLI args populate a [dwave] section.
    dwave_cfg = dict(conf.get("dwave") or {})

    if dwave_api_key is not None:
        dwave_cfg["token"] = dwave_api_key
        os.environ["DWAVE_API_KEY"] = dwave_api_key
        os.environ["DWAVE_API_TOKEN"] = dwave_api_key
    if dwave_api_solver is not None:
        dwave_cfg["solver"] = dwave_api_solver
        os.environ["DWAVE_API_SOLVER"] = dwave_api_solver
    if dwave_region_url is not None:
        dwave_cfg["dwave_region_url"] = dwave_region_url
        os.environ["DWAVE_REGION_URL"] = dwave_region_url
    if qpu_daily_budget is not None:
        dwave_cfg["daily_budget"] = qpu_daily_budget

    # Ensure env vars are set from TOML
    _set_dwave_env(dwave_cfg)

    conf["dwave"] = dwave_cfg
    # Ensure [qpu] exists so _initialize_miners detects QPU mode
    if "qpu" not in conf:
        conf["qpu"] = {}

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

    def _cmd_for(kind: str, port: int, peer: Optional[str], rest_port: int) -> list[str]:
        # Pin both listen and public-host to 127.0.0.1: avoids public-IP
        # auto-detection (which causes children to advertise themselves on
        # the host's NAT'd public address and then fail JOINs against real
        # mainnet peers from the genesis config).
        base = [
            "quip-network-node", kind,
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--public-host", "127.0.0.1",
            "--rest-insecure-port", str(rest_port),
        ]
        if peer:
            base += ["--peer", peer]
        return base

    # Assign each child a unique REST HTTP port so they don't collide on
    # the default 20050. Offset by index so the mapping is stable: the
    # child on QUIC port base_port + i gets REST on 20050 + i.
    rest_base = 20050

    # Build command list
    peer_addr = None
    child_index = 0
    for kind, count in order:
        for _ in range(count):
            rest_port = rest_base + child_index
            if kind == bootstrap_kind and peer_addr is None:
                cmds.append(_cmd_for(kind, port, None, rest_port))
                # Use the loopback literal so name resolution doesn't
                # accidentally pick ::1 before the v4 listener binds.
                peer_addr = f"127.0.0.1:{port}"
            else:
                cmds.append(_cmd_for(kind, port, peer_addr, rest_port))
            port += 1
            child_index += 1

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

def network_simulator_main():
    # Set multiprocessing start method to 'spawn' to avoid context mixing issues
    multiprocessing.set_start_method('spawn', force=True)

    quip_network_simulator(standalone_mode=False)

