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
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List

import click
import traceback


# TOML loader supporting Python 3.10 via tomli and 3.11+ via tomllib
try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore

from shared.block import load_genesis_block
from shared.keystore import generate
from shared.logging_config import setup_logging
from shared.miner_bootstrap import BootstrapConfig, bootstrap
from shared.network_node import NetworkNode
from shared.version import get_version


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
        "cuda",
        "nvidia",
        "metal",
        "modal",
        "dwave",
        "ibm",
        "braket",
        "pasqal",
        "ionq",
        "origin",
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
        out[k] = v
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


def _apply_global_overrides(
    conf: Dict[str, Any],
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
    rest_insecure_port: Optional[int] = None,
) -> Dict[str, Any]:
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


async def _async_run_network_node(
    config: Dict[str, Any], genesis_config_file: str
) -> int:
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
        node_name=node_name,
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
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to TOML config file",
)
@click.option("--version", is_flag=True, help="Show version and exit")
@click.option("--debug-config", is_flag=True, help="Print final configuration as JSON")
@click.pass_context
def quip_network_node(
    ctx: click.Context, config: Optional[str], version: bool, debug_config: bool
):
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
            raise click.UsageError(
                "No subcommand given and no miner sections ([cpu], [gpu], [qpu]) found in config"
            )

        # Apply debug config from global options
        if ctx.obj.get("debug_config", False):
            _print_final_config(cfg, "auto-configured")

        # Use genesis_block.json as default genesis config
        genesis_config = cfg.get("genesis_config", "genesis_block.json")

        sys.exit(_run_network_node_sync(cfg, genesis_config))


# Subcommands: cpu/gpu/qpu. Each builds a NetworkNode config from TOML and CLI flags.


@quip_network_node.command(name="cpu")
# Global network options
@click.option(
    "--listen",
    type=str,
    default=None,
    help="Address to bind; IPv6 supported (e.g., ::1 or ::). Defaults from [global].listen or 127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind (defaults from [global].port or 20049)",
)
@click.option(
    "--public-host",
    type=str,
    default=None,
    help="Public hostname or IP advertised to peers",
)
@click.option(
    "--public-port",
    type=int,
    default=None,
    help="Public port advertised to peers (defaults to --port)",
)
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option(
    "--secret", type=str, default=None, help="Deterministic secret for keypair"
)
@click.option(
    "--auto-mine/--no-auto-mine",
    default=None,
    help="Enable/disable auto-mining when no peers found",
)
@click.option(
    "--peer",
    "peers",
    multiple=True,
    help="Peer host:port; use [IPv6]:port for IPv6 (repeat for multiple)",
)
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option(
    "--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats"
)
@click.option(
    "--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds"
)
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging level",
)
@click.option(
    "--node-log",
    type=str,
    default=None,
    help="Path to main node log file (defaults to stderr)",
)
@click.option(
    "--http-log",
    type=str,
    default=None,
    help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)",
)
# REST API options
@click.option(
    "--rest-port",
    type=int,
    default=None,
    help="REST API HTTPS port (-1 disables, defaults from [global].rest_port or -1)",
)
@click.option(
    "--rest-insecure-port",
    type=int,
    default=None,
    help="REST API HTTP port (-1 disables, defaults from [global].rest_insecure_port or 20050)",
)
# CPU options
@click.option(
    "--num-cpus",
    type=int,
    default=None,
    help="Number of CPU miners to spawn (default 1)",
)
# Other
@click.option(
    "--genesis-config",
    type=str,
    default="genesis_block.json",
    show_default=True,
    help="Genesis block configuration file",
)
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

    conf = _apply_global_overrides(
        conf,
        listen,
        port,
        public_host,
        public_port,
        node_name,
        secret,
        auto_mine,
        list(peers) or None,
        timeout,
        heartbeat_interval,
        heartbeat_timeout,
        fanout,
        log_level,
        node_log,
        http_log,
        rest_port,
        rest_insecure_port,
    )

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
@click.option(
    "--listen",
    type=str,
    default=None,
    help="Address to bind; IPv6 supported (e.g., ::1 or ::). Defaults from [global].listen or 127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind (defaults from [global].port or 20049)",
)
@click.option(
    "--public-host",
    type=str,
    default=None,
    help="Public hostname or IP advertised to peers",
)
@click.option(
    "--public-port",
    type=int,
    default=None,
    help="Public port advertised to peers (defaults to --port)",
)
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option(
    "--secret", type=str, default=None, help="Deterministic secret for keypair"
)
@click.option(
    "--auto-mine/--no-auto-mine",
    default=None,
    help="Enable/disable auto-mining when no peers found",
)
@click.option(
    "--peer",
    "peers",
    multiple=True,
    help="Peer host:port; use [IPv6]:port for IPv6 (repeat for multiple)",
)
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option(
    "--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats"
)
@click.option(
    "--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds"
)
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging level",
)
@click.option(
    "--node-log",
    type=str,
    default=None,
    help="Path to main node log file (defaults to stderr)",
)
@click.option(
    "--http-log",
    type=str,
    default=None,
    help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)",
)
# REST API options
@click.option(
    "--rest-port",
    type=int,
    default=None,
    help="REST API HTTPS port (-1 disables, defaults from [global].rest_port or -1)",
)
@click.option(
    "--rest-insecure-port",
    type=int,
    default=None,
    help="REST API HTTP port (-1 disables, defaults from [global].rest_insecure_port or 20050)",
)
# GPU options
@click.option(
    "--gpu-backend",
    type=click.Choice(["local", "modal", "mps"], case_sensitive=False),
    default=None,
    help="GPU backend: local|modal|mps",
)
@click.option(
    "--device",
    "devices",
    multiple=True,
    help="GPU device(s) for local backend (e.g., 0 1)",
)
@click.option(
    "--gpu-type",
    "gpu_types",
    multiple=True,
    help="GPU type(s) for modal backend (e.g., t4 a10g)",
)
@click.option(
    "--gpu-utilization",
    type=int,
    default=100,
    help="GPU utilization percentage (1-100, default: 100)",
)
@click.option(
    "--yielding",
    is_flag=True,
    default=False,
    help="Yield GPU to other processes (NVML-adaptive nonce scaling)",
)
# Other
@click.option(
    "--genesis-config",
    type=str,
    default="genesis_block.json",
    show_default=True,
    help="Genesis block configuration file",
)
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
    conf = _apply_global_overrides(
        conf,
        listen,
        port,
        public_host,
        public_port,
        node_name,
        secret,
        auto_mine,
        list(peers) or None,
        timeout,
        heartbeat_interval,
        heartbeat_timeout,
        fanout,
        log_level,
        node_log,
        http_log,
        rest_port,
        rest_insecure_port,
    )

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
@click.option(
    "--listen",
    type=str,
    default=None,
    help="Address to bind; IPv6 supported (e.g., ::1 or ::). Defaults from [global].listen or 127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind (defaults from [global].port or 20049)",
)
@click.option(
    "--public-host",
    type=str,
    default=None,
    help="Public hostname or IP advertised to peers",
)
@click.option(
    "--public-port",
    type=int,
    default=None,
    help="Public port advertised to peers (defaults to --port)",
)
@click.option("--node-name", type=str, default=None, help="Human-readable node name")
@click.option(
    "--secret", type=str, default=None, help="Deterministic secret for keypair"
)
@click.option(
    "--auto-mine/--no-auto-mine",
    default=None,
    help="Enable/disable auto-mining when no peers found",
)
@click.option(
    "--peer",
    "peers",
    multiple=True,
    help="Peer host:port; use [IPv6]:port for IPv6 (repeat for multiple)",
)
@click.option("--timeout", type=int, default=None, help="Node/network timeout seconds")
@click.option(
    "--heartbeat-interval", type=int, default=None, help="Seconds between heartbeats"
)
@click.option(
    "--heartbeat-timeout", type=int, default=None, help="Peer heartbeat timeout seconds"
)
@click.option("--fanout", type=int, default=None, help="Gossip fanout")
# Logging options
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging level",
)
@click.option(
    "--node-log",
    type=str,
    default=None,
    help="Path to main node log file (defaults to stderr)",
)
@click.option(
    "--http-log",
    type=str,
    default=None,
    help="Path to HTTP log file or 'stderr'/'stdout' for console (suppresses aiohttp logs if not set)",
)
# REST API options
@click.option(
    "--rest-port",
    type=int,
    default=None,
    help="REST API HTTPS port (-1 disables, defaults from [global].rest_port or -1)",
)
@click.option(
    "--rest-insecure-port",
    type=int,
    default=None,
    help="REST API HTTP port (-1 disables, defaults from [global].rest_insecure_port or 20050)",
)
# QPU options
@click.option("--dwave-api-key", type=str, default=None, help="D-Wave API key")
@click.option("--dwave-api-solver", type=str, default=None, help="D-Wave solver name")
@click.option(
    "--dwave-region-url", type=str, default=None, help="D-Wave SAPI region endpoint URL"
)
# QPU time budget options
@click.option(
    "--qpu-daily-budget",
    type=str,
    default=None,
    help="Daily QPU time budget (e.g., 40s, 2m) - calculate from your Leap allocation",
)
# Other
@click.option(
    "--genesis-config",
    type=str,
    default="genesis_block.json",
    show_default=True,
    help="Genesis block configuration file",
)
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
    conf = _apply_global_overrides(
        conf,
        listen,
        port,
        public_host,
        public_port,
        node_name,
        secret,
        auto_mine,
        list(peers) or None,
        timeout,
        heartbeat_interval,
        heartbeat_timeout,
        fanout,
        log_level,
        node_log,
        http_log,
        rest_port,
        rest_insecure_port,
    )

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
@click.option(
    "--scenario",
    type=click.Choice(["mixed", "cpu", "gpu"], case_sensitive=False),
    default="mixed",
    show_default=True,
    help="Network scenario to launch",
)
@click.option("--num-cpu", type=int, default=None, help="Override: number of CPU nodes")
@click.option("--num-gpu", type=int, default=None, help="Override: number of GPU nodes")
@click.option("--num-qpu", type=int, default=None, help="Override: number of QPU nodes")
@click.option(
    "--base-port",
    type=int,
    default=8080,
    show_default=True,
    help="Starting port for first node",
)
@click.option("--print-only", is_flag=True, help="Only print commands, do not execute")
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def quip_network_simulator(
    ctx: click.Context,
    scenario: str,
    num_cpu: Optional[int],
    num_gpu: Optional[int],
    num_qpu: Optional[int],
    base_port: int,
    print_only: bool,
    version: bool,
):
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

    def _cmd_for(
        kind: str, port: int, peer: Optional[str], rest_port: int
    ) -> list[str]:
        # Pin both listen and public-host to 127.0.0.1: avoids public-IP
        # auto-detection (which causes children to advertise themselves on
        # the host's NAT'd public address and then fail JOINs against real
        # mainnet peers from the genesis config).
        base = [
            "quip-network-node",
            kind,
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--public-host",
            "127.0.0.1",
            "--rest-insecure-port",
            str(rest_port),
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
@click.argument(
    "target",
    type=click.Choice(
        ["cpu", "gpu-local", "gpu-metal", "gpu-modal", "qpu"], case_sensitive=False
    ),
)
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


# ---------------------------------------------------------------------------
# quip-miner CLI (Phase 2: bootstrap + faucet-bot + keygen).
#
# Substrate-integrated mining frontend. Replaces the legacy P2P-driven
# `quip-network-node` flow over the v0.1 → v0.2 refactor. Per the integration
# plan, the cpu/gpu/qpu mining subcommands themselves are added in Phase 4
# alongside the SubstrateMinerController; Phase 2 only ships the bootstrap
# tooling so an operator can fund + register a miner account.
# ---------------------------------------------------------------------------


@click.group(name="quip-miner")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level for quip-miner subcommands",
)
def quip_miner(log_level: str) -> None:
    """Substrate-integrated quantum mining frontend."""
    setup_logging(log_level=log_level.upper(), node_name="quip-miner")


@quip_miner.command("keygen")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help="Where to write the generated keystore",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing keystore at --out")
def quip_miner_keygen(out_path: str, overwrite: bool) -> None:
    """Generate a fresh sr25519 signing key for quip-miner.

    Writes a JSON keystore (0o600) and prints the SS58 address. The seed is
    stored in plaintext — adequate for dev workflows where the faucet bot
    runs alongside. Passphrase-encrypted keystores land in Phase 7.
    """
    keystore = generate(Path(out_path).expanduser(), overwrite=overwrite)
    click.echo(f"wrote keystore: {keystore.path}")
    click.echo(f"ss58 address:   {keystore.signer.ss58_address()}")
    click.echo(f"account_id:     0x{keystore.signer.account_id_bytes().hex()}")


@quip_miner.command("bootstrap")
@click.option(
    "--node-url",
    required=True,
    help="Substrate node WebSocket URL (e.g. ws://localhost:9944)",
)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help="Path to the keystore (will be created if missing)",
)
@click.option(
    "--faucet-url",
    default=None,
    help="If set, request funding from this faucet bot when balance is low",
)
@click.option(
    "--seed-chain/--no-seed-chain",
    default=False,
    help="DEV-ONLY: on a fresh dev chain (Development / Local Testnet / "
    "quip-local), sudo-seed QuantumPow.Difficulty and "
    "QuantumPow.DefaultTopology. Refused against non-dev chains; never "
    "used in production (default off; requires --sudo-key)",
)
@click.option(
    "--sudo-key",
    "sudo_key_uri",
    default="//Alice",
    show_default=True,
    help="Substrate URI for the sudo signer (dev only)",
)
@click.option(
    "--seed-topology",
    "seed_topology_mt",
    default="2,2",
    show_default=True,
    help="Zephyr Z(m,t) parameters for the seed topology when --seed-chain is set",
)
def quip_miner_bootstrap(
    node_url: str,
    signer_key_path: str,
    faucet_url: Optional[str],
    seed_chain: bool,
    sudo_key_uri: str,
    seed_topology_mt: str,
) -> None:
    """Fund and register a miner account against a substrate chain.

    Idempotent — re-running against a fully-bootstrapped account is a no-op
    that just verifies state.
    """
    try:
        m_str, t_str = seed_topology_mt.split(",")
        topology_mt = (int(m_str), int(t_str))
    except (ValueError, AttributeError) as exc:
        raise click.BadParameter(
            f"--seed-topology must be 'm,t' (e.g. '2,2'), got {seed_topology_mt!r}"
        ) from exc

    config = BootstrapConfig(
        node_url=node_url,
        signer_key_path=Path(signer_key_path).expanduser(),
        faucet_url=faucet_url,
        sudo_key_uri=sudo_key_uri,
        seed_chain=seed_chain,
        seed_topology_mt=topology_mt,
    )
    result = asyncio.run(bootstrap(config))

    click.echo("bootstrap complete")
    click.echo(f"  ss58 address       : {result.ss58_address}")
    click.echo(f"  account_id         : {result.account_id_hex}")
    click.echo(f"  balance (plancks)  : {result.balance_plancks}")
    click.echo(f"  miner registered   : {result.miner_registered}")
    click.echo(f"  topology seeded    : {result.topology_seeded}")
    click.echo(f"  difficulty seeded  : {result.difficulty_seeded}")


# The faucet bot is shipped as a standalone script (`faucet_bot.py` at repo
# root) so it can be deployed independently of the rest of `quip-protocol`.
# Run with:
#
#     python faucet_bot.py --node-url ws://localhost:9944 --faucet-key //Alice
#
# See faucet_bot.py for the full CLI surface.


# --------------------------------------------------------------------------
# Mining subcommands: cpu / gpu / qpu spawn the controller end-to-end.
# Topology binding is explicit via --topology zephyr:M,T (defaults to Z(9,2)
# which matches the legacy chain difficulty calibration). The CLI verifies
# at startup that the sampler's topology hash matches the chain's snapshot;
# mismatch fails fast with the registered hash printed so the operator can
# fix --topology or re-seed the chain.
# --------------------------------------------------------------------------


def _parse_topology(spec: str):
    """Parse a `family:m,t` topology spec and return the topology object.

    Only zephyr is supported in Phase 5a; pegasus/chimera land later if
    miners need them. Returns the sampler-compatible topology object so the
    CLI can plug it straight into the spec's `args["topology"]`.
    """
    from dwave_topologies.topologies.zephyr import zephyr

    if ":" not in spec:
        raise click.BadParameter(
            f"--topology must be 'family:m,t' (got {spec!r}); try 'zephyr:9,2'"
        )
    family, params = spec.split(":", 1)
    if family.lower() != "zephyr":
        raise click.BadParameter(
            f"only 'zephyr' topology is supported in Phase 5a (got {family!r})"
        )
    try:
        m_str, t_str = params.split(",")
        m, t = int(m_str), int(t_str)
        return zephyr(m, t)
    except (ValueError, Exception) as exc:
        raise click.BadParameter(
            f"--topology zephyr:{params!r} is invalid: {exc}"
        ) from exc


async def _run_miner(
    *,
    kind: str,
    node_url: str,
    signer_key_path: str,
    rest_port: int,
    topology_spec: str,
    miner_config: dict,
):
    """Shared entry point for `quip-miner cpu|gpu|qpu`.

    Builds the keystore-loaded signer, a `MinerCore` with the requested
    miner kind, two `SubstrateClient` instances (state + subscription),
    and a `SubstrateMinerController`. Runs the controller until KeyboardInterrupt.
    """
    import asyncio
    import signal as signal_module
    from pathlib import Path

    from shared.keystore import load
    from shared.miner_core import MinerCore
    from shared.substrate_client import SubstrateClient
    from shared.substrate_miner_controller import SubstrateMinerController

    keystore = load(Path(signer_key_path).expanduser())
    click.echo(f"signer: {keystore.signer.ss58_address()}")

    # Build the sampler topology and bind it to the miner config.
    topology = _parse_topology(topology_spec)
    click.echo(
        f"topology: {topology_spec} ({topology.num_nodes} nodes, "
        f"{topology.num_edges} edges)"
    )

    # Inject the topology into whichever miner kind is being constructed.
    miner_config = _inject_topology(miner_config, kind, topology)

    core = MinerCore(node_id="quip-miner", miners_config=miner_config)
    try:
        if not core.miner_handles:
            click.echo(
                f"no miner handles built for kind={kind}; check --num-cpus / "
                f"GPU/QPU config",
                err=True,
            )
            return 2

        client = SubstrateClient(url=node_url)
        await client.connect()
        try:
            # Verify the sampler topology matches the chain's registered topology
            # before mining starts. Without this check, every proof would be rejected
            # with InvalidTopology (Phase 4 confirmed this end-to-end).
            head = await client.get_head()
            snapshot = await client.get_mining_snapshot(
                at=head,
                miner_account_bytes=keystore.signer.account_id_bytes(),
            )
            if snapshot is None:
                click.echo(
                    "chain has no registered topology; run "
                    "`quip-miner bootstrap --seed-chain` first",
                    err=True,
                )
                return 3
            expected_hash = _zephyr_topology_hash(topology)
            if snapshot.topology_hash != expected_hash:
                click.echo(
                    f"topology mismatch: --topology {topology_spec} hashes to "
                    f"0x{expected_hash.hex()} but chain has 0x{snapshot.topology_hash.hex()}; "
                    "either adjust --topology or re-seed the chain",
                    err=True,
                )
                return 4

            controller = SubstrateMinerController(
                client=client,
                signer=keystore.signer,
                miner_handles=core.miner_handles,
                topology_hash=snapshot.topology_hash,
            )
            click.echo(
                f"controller starting: handles={[h.miner_id for h in core.miner_handles]} "
                f"topology_hash=0x{snapshot.topology_hash.hex()[:16]}..."
            )

            loop = asyncio.get_running_loop()
            for sig in (signal_module.SIGINT, signal_module.SIGTERM):
                loop.add_signal_handler(sig, controller.shutdown)

            await controller.run()
            click.echo(f"controller stopped: stats={controller.stats}")
        finally:
            await client.close()
    finally:
        core.close()
    return 0


def _inject_topology(miner_config: dict, kind: str, topology) -> dict:
    """Stash the topology in the appropriate miner section's `args`.

    The miner-builder paths in `MinerCore._initialize_miners` forward
    `args["topology"]` straight to the sampler constructor.
    """
    out = dict(miner_config)
    if kind == "cpu":
        section = dict(out.get("cpu") or {})
        args = dict(section.get("args") or {})
        args["topology"] = topology
        section["args"] = args
        out["cpu"] = section
    else:
        # GPU/QPU sections are device-array shaped; topology injection happens
        # at the spec level via _build_gpu_specs / _build_qpu_specs in a
        # follow-on once GPU/QPU end-to-end is exercised against the chain.
        click.echo(
            f"warning: --topology injection not yet implemented for kind={kind!r}; "
            "GPU/QPU samplers will use their default topology (Phase 6)",
            err=True,
        )
    return out


def _zephyr_topology_hash(topology) -> bytes:
    """Compute the chain's blake2_256 topology hash from a Zephyr graph.

    Matches `pallets/quantum-pow/src/topology.rs::hash_topology`:
        blake2_256(SCALE((sorted nodes, sorted canonical edges)))
    """
    import hashlib

    nodes = sorted(int(n) for n in topology.nodes)
    edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in topology.edges
    )

    def compact_len(n: int) -> bytes:
        if n < 0x40:
            return bytes([n << 2])
        if n < 0x4000:
            return ((n << 2) | 0b01).to_bytes(2, "little")
        if n < 0x40000000:
            return ((n << 2) | 0b10).to_bytes(4, "little")
        raise ValueError(f"compact len {n} exceeds 30-bit range")

    buf = compact_len(len(nodes))
    for n in nodes:
        buf += n.to_bytes(4, "little")
    buf += compact_len(len(edges))
    for u, v in edges:
        buf += u.to_bytes(4, "little") + v.to_bytes(4, "little")
    return hashlib.blake2b(buf, digest_size=32).digest()


_NODE_URL_HELP = "Substrate node WebSocket URL (e.g. ws://localhost:9944)"
_SIGNER_KEY_HELP = "Path to the signing keystore (created by `quip-miner keygen`)"
_TOPOLOGY_HELP = (
    "Topology spec for the miner's sampler. Format: 'family:m,t'. Must hash "
    "to the chain's registered topology — mismatch fails fast at startup."
)
_REST_PORT_HELP = (
    "Telemetry REST API port (set to -1 to disable; Phase 5b wires this in)"
)


@quip_miner.command("cpu")
@click.option("--node-url", required=True, help=_NODE_URL_HELP)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help=_SIGNER_KEY_HELP,
)
@click.option(
    "--num-cpus",
    type=int,
    default=1,
    show_default=True,
    help="Number of CPU SA workers to spawn",
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=-1,
    show_default=True,
    help=_REST_PORT_HELP,
)
def quip_miner_cpu(
    node_url: str,
    signer_key_path: str,
    num_cpus: int,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run CPU SA miners against a substrate chain."""
    import asyncio

    miner_config = {"cpu": {"num_cpus": num_cpus}}
    raise SystemExit(
        asyncio.run(
            _run_miner(
                kind="cpu",
                node_url=node_url,
                signer_key_path=signer_key_path,
                rest_port=rest_port,
                topology_spec=topology_spec,
                miner_config=miner_config,
            )
        )
    )


@quip_miner.command("gpu")
@click.option("--node-url", required=True, help=_NODE_URL_HELP)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help=_SIGNER_KEY_HELP,
)
@click.option(
    "--gpu-backend",
    type=click.Choice(["local", "metal", "modal"], case_sensitive=False),
    default="local",
    show_default=True,
    help="GPU backend: local CUDA, Apple Metal, or Modal cloud",
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=-1,
    show_default=True,
    help=_REST_PORT_HELP,
)
def quip_miner_gpu(
    node_url: str,
    signer_key_path: str,
    gpu_backend: str,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run a GPU miner (CUDA / Metal / Modal) against a substrate chain.

    GPU end-to-end verification against the chain is a Phase 6 follow-on —
    the controller and spec wiring work, but `--topology` injection for
    GPU samplers landed in a single CPU-only path in Phase 5a. CUDA / Metal
    samplers consume the topology via their `args` dict the same way the
    CPU path does once that injection is generalised.
    """
    import asyncio

    backend = gpu_backend.lower()
    if backend == "local":
        miner_config = {"cuda": [{"device": "0"}]}
    elif backend == "metal":
        miner_config = {"metal": [{}]}
    elif backend == "modal":
        miner_config = {"modal": [{"gpu_type": "t4"}]}
    else:
        raise click.BadParameter(f"unknown --gpu-backend: {backend}")

    raise SystemExit(
        asyncio.run(
            _run_miner(
                kind="gpu",
                node_url=node_url,
                signer_key_path=signer_key_path,
                rest_port=rest_port,
                topology_spec=topology_spec,
                miner_config=miner_config,
            )
        )
    )


@quip_miner.command("qpu")
@click.option("--node-url", required=True, help=_NODE_URL_HELP)
@click.option(
    "--signer-key",
    "signer_key_path",
    type=click.Path(dir_okay=False),
    default="~/.quip-miner/signing.json",
    show_default=True,
    help=_SIGNER_KEY_HELP,
)
@click.option(
    "--qpu-type",
    type=click.Choice(["dwave", "ibm", "ionq", "pasqal", "braket", "origin"]),
    default="dwave",
    show_default=True,
    help="QPU provider",
)
@click.option(
    "--daily-budget",
    type=str,
    default=None,
    help=(
        "Daily QPU access-time budget. Format: '30s', '5m', '2h', '1d'. "
        "Passed to QPUTimeManager; omit to run without a daily cap."
    ),
)
@click.option(
    "--topology",
    "topology_spec",
    default="zephyr:9,2",
    show_default=True,
    help=_TOPOLOGY_HELP,
)
@click.option(
    "--rest-port",
    type=int,
    default=-1,
    show_default=True,
    help=_REST_PORT_HELP,
)
def quip_miner_qpu(
    node_url: str,
    signer_key_path: str,
    qpu_type: str,
    daily_budget,
    topology_spec: str,
    rest_port: int,
) -> None:
    """Run a QPU miner against a substrate chain.

    Provider credentials come from the environment (e.g. DWAVE_API_KEY).
    Same Phase 5a caveat as GPU: end-to-end against the chain is a Phase 6
    item once topology binding generalises beyond CPU.
    """
    import asyncio

    section: dict = {"type": qpu_type}
    if daily_budget is not None:
        section["daily_budget"] = daily_budget
    miner_config = {qpu_type: [section]}

    raise SystemExit(
        asyncio.run(
            _run_miner(
                kind="qpu",
                node_url=node_url,
                signer_key_path=signer_key_path,
                rest_port=rest_port,
                topology_spec=topology_spec,
                miner_config=miner_config,
            )
        )
    )


# Entry points for console_scripts


def network_node_main():
    # Set multiprocessing start method to 'spawn' to avoid context mixing issues
    multiprocessing.set_start_method("spawn", force=True)

    quip_network_node(standalone_mode=False)


def network_simulator_main():
    # Set multiprocessing start method to 'spawn' to avoid context mixing issues
    multiprocessing.set_start_method("spawn", force=True)

    quip_network_simulator(standalone_mode=False)


def miner_main():
    """Entry point for the `quip-miner` console script."""
    try:
        quip_miner(standalone_mode=False)
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(f"quip-miner: error: {exc}", err=True)
        raise SystemExit(1) from exc
