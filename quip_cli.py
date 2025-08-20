"""Click-based CLI for quip-protocol.

Provides two console commands:
- quip-network-node: run a single P2P node (cpu/gpu/qpu) using quantum_blockchain_p2p.py
- quip-network-simulator: launch multiple nodes using quip-network-node and connect them locally to each other
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any, Dict, Optional

import click

# TOML loader supporting Python 3.10 via tomli and 3.11+ via tomllib
try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "rb") as f:
        return _toml.load(f)


# -----------------------------
# quip-network-node
# -----------------------------

@click.group(invoke_without_command=True)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), help="Path to TOML config file")
@click.pass_context
def quip_network_node(ctx: click.Context, config: Optional[str]):
    """Run a single quip network node.

    Subcommands: cpu, gpu, qpu

    If invoked without a subcommand, --config must specify [global].default
    to choose the default subcommand. Global settings also provide host/port/peer/auto_mine.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)

    if ctx.invoked_subcommand is None:
        cfg = ctx.obj.get("config", {})
        global_cfg = (cfg.get("global", {}) or {})
        default_cmd = (global_cfg or {}).get("default")
        if not default_cmd:
            raise click.UsageError("No subcommand given and no [global].default in config")
        default_cmd = str(default_cmd).lower()
        if default_cmd not in {"cpu", "gpu", "qpu"}:
            raise click.UsageError(f"Invalid default subcommand '{default_cmd}' in config; expected cpu/gpu/qpu")
        # Build common args from [global]
        common_kwargs = {
            "host": global_cfg.get("host", "0.0.0.0"),
            "port": int(global_cfg.get("port", 8080)),
            "peer": global_cfg.get("peer"),
            "auto_mine": int(global_cfg.get("auto_mine", 0)),
        }
        # Optional per-type config
        sub_cfg = cfg.get(default_cmd, {}) or {}
        if default_cmd == "cpu":
            ctx.invoke(cpu, num_cpus=sub_cfg.get("num_cpus"), **common_kwargs)
        elif default_cmd == "gpu":
            devices = sub_cfg.get("devices") or []
            device = devices[0] if isinstance(devices, list) and devices else None
            ctx.invoke(gpu, device=device, **common_kwargs)
        else:
            qpu_kwargs = {
                "dwave_api_key": sub_cfg.get("dwave_api_key"),
                "dwave_api_solver": sub_cfg.get("dwave_api_solver"),
                "dwave_region_url": sub_cfg.get("dwave_region_url", "https://na-west-1.cloud.dwavesys.com/sapi/v2/"),
            }
            ctx.invoke(qpu, **common_kwargs, **qpu_kwargs)


def _run_p2p_node(
    kind: str,
    host: str,
    port: int,
    peer: Optional[str],
    auto_mine: int,
    env_overrides: Optional[dict] = None,
) -> int:
    """Run quantum_blockchain_p2p.py as a subprocess for a single node of given kind.

    kind: one of 'cpu', 'gpu', 'qpu'
    """
    script = os.path.join(os.path.dirname(__file__), "quantum_blockchain_p2p.py")
    cmd = [
        sys.executable,
        script,
        "--host",
        host,
        "--port",
        str(port),
        "--competitive",
    ]
    if kind == "cpu":
        cmd += ["--num-sa", "1"]
    elif kind == "gpu":
        cmd += ["--num-gpu", "1"]
    elif kind == "qpu":
        cmd += ["--num-qpu", "1"]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if peer:
        cmd += ["--peer", peer]
    if auto_mine and int(auto_mine) > 0:
        cmd += ["--auto-mine", str(int(auto_mine))]

    click.echo(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items() if v is not None})
    return subprocess.call(cmd, env=env)


@quip_network_node.command(name="cpu")
@click.option("--host", type=str, default=None, help="Host to bind to (defaults from [global].host or 0.0.0.0)")
@click.option("--port", type=int, default=None, help="Port to bind to (defaults from [global].port or 8080)")
@click.option("--peer", type=str, default=None, help="Peer address host:port to join (defaults from [global].peer)")
@click.option("--auto-mine", type=int, default=None, help="Automatically mine N blocks (defaults from [global].auto_mine or 0)")
@click.option("--num-cpus", type=int, help="Limit CPU threads via OMP/MKL/BLAS env vars")
@click.pass_context
def cpu(ctx: click.Context, host: Optional[str], port: Optional[int], peer: Optional[str], auto_mine: Optional[int], num_cpus: Optional[int]):
    """Run a CPU node (1 SA miner)."""
    global_cfg = ((ctx.obj or {}).get("config", {}) or {}).get("global", {})
    host = host if host is not None else global_cfg.get("host", "0.0.0.0")
    port = port if port is not None else int(global_cfg.get("port", 8080))
    peer = peer if peer is not None else global_cfg.get("peer")
    auto_mine = auto_mine if auto_mine is not None else int(global_cfg.get("auto_mine", 0))

    env = None
    if num_cpus:
        env = {
            "OMP_NUM_THREADS": num_cpus,
            "MKL_NUM_THREADS": num_cpus,
            "OPENBLAS_NUM_THREADS": num_cpus,
            "NUMEXPR_NUM_THREADS": num_cpus,
        }
    sys.exit(_run_p2p_node("cpu", host, port, peer, auto_mine, env_overrides=env))


@quip_network_node.command(name="gpu")
@click.option("--host", type=str, default=None, help="Host to bind to (defaults from [global].host or 0.0.0.0)")
@click.option("--port", type=int, default=None, help="Port to bind to (defaults from [global].port or 8080)")
@click.option("--peer", type=str, default=None, help="Peer address host:port to join (defaults from [global].peer)")
@click.option("--auto-mine", type=int, default=None, help="Automatically mine N blocks (defaults from [global].auto_mine or 0)")
@click.option("--device", type=str, help="GPU device selector (e.g., CUDA ordinal)")
@click.pass_context
def gpu(ctx: click.Context, host: Optional[str], port: Optional[int], peer: Optional[str], auto_mine: Optional[int], device: Optional[str]):
    """Run a GPU node (1 GPU miner)."""
    global_cfg = ((ctx.obj or {}).get("config", {}) or {}).get("global", {})
    host = host if host is not None else global_cfg.get("host", "0.0.0.0")
    port = port if port is not None else int(global_cfg.get("port", 8080))
    peer = peer if peer is not None else global_cfg.get("peer")
    auto_mine = auto_mine if auto_mine is not None else int(global_cfg.get("auto_mine", 0))

    # If config provided devices array and device not given, pick first
    cfg = (ctx.obj or {}).get("config", {}) if hasattr(ctx, "obj") else {}
    if device is None:
        devs = (cfg.get("gpu", {}) or {}).get("devices")
        if isinstance(devs, list) and devs:
            device = str(devs[0])
    env = {"CUDA_VISIBLE_DEVICES": device} if device else None
    sys.exit(_run_p2p_node("gpu", host, port, peer, auto_mine, env_overrides=env))


@quip_network_node.command(name="qpu")
@click.option("--host", type=str, default=None, help="Host to bind to (defaults from [global].host or 0.0.0.0)")
@click.option("--port", type=int, default=None, help="Port to bind to (defaults from [global].port or 8080)")
@click.option("--peer", type=str, default=None, help="Peer address host:port to join (defaults from [global].peer)")
@click.option("--auto-mine", type=int, default=None, help="Automatically mine N blocks (defaults from [global].auto_mine or 0)")
@click.option("--dwave-api-key", type=str, help="D-Wave API key (DWAVE_API_TOKEN)")
@click.option("--dwave-api-solver", type=str, help="D-Wave solver name (DWAVE_API_SOLVER)")
@click.option("--dwave-region-url", type=str, default=None, help="D-Wave region SAPI endpoint URL")
@click.pass_context
def qpu(
    ctx: click.Context,
    host: Optional[str],
    port: Optional[int],
    peer: Optional[str],
    auto_mine: Optional[int],
    dwave_api_key: Optional[str],
    dwave_api_solver: Optional[str],
    dwave_region_url: Optional[str],
):
    """Run a QPU node (1 QPU miner)."""
    global_cfg = ((ctx.obj or {}).get("config", {}) or {}).get("global", {})
    host = host if host is not None else global_cfg.get("host", "0.0.0.0")
    port = port if port is not None else int(global_cfg.get("port", 8080))
    peer = peer if peer is not None else global_cfg.get("peer")
    auto_mine = auto_mine if auto_mine is not None else int(global_cfg.get("auto_mine", 0))

    # Fill from config if not provided
    cfg = (ctx.obj or {}).get("config", {}) if hasattr(ctx, "obj") else {}
    qpu_cfg = (cfg.get("qpu", {}) or {})
    if not dwave_api_key:
        dwave_api_key = qpu_cfg.get("dwave_api_key")
    if not dwave_api_solver:
        dwave_api_solver = qpu_cfg.get("dwave_api_solver")
    if not dwave_region_url:
        dwave_region_url = qpu_cfg.get("dwave_region_url", "https://na-west-1.cloud.dwavesys.com/sapi/v2/")

    env = {
        "DWAVE_API_TOKEN": dwave_api_key,
        "DWAVE_API_SOLVER": dwave_api_solver,
        "DWAVE_API_ENDPOINT": dwave_region_url,
    }
    # Remove None values
    env = {k: v for k, v in env.items() if v}
    sys.exit(_run_p2p_node("qpu", host, port, peer, auto_mine, env_overrides=env))


# -----------------------------
# quip-network-simulator
# -----------------------------

@click.command(name="quip-network-simulator")
@click.option("--scenario", type=click.Choice(["mixed", "cpu", "gpu"], case_sensitive=False), default="mixed", show_default=True, help="Network scenario to launch")
@click.option("--num-cpu", type=int, default=None, help="Override: number of CPU nodes")
@click.option("--num-gpu", type=int, default=None, help="Override: number of GPU nodes")
@click.option("--num-qpu", type=int, default=None, help="Override: number of QPU nodes")
@click.option("--base-port", type=int, default=8080, show_default=True, help="Starting port for first node")
@click.option("--print-only", is_flag=True, help="Only print commands, do not execute")
def quip_network_simulator(scenario: str, num_cpu: Optional[int], num_gpu: Optional[int], num_qpu: Optional[int], base_port: int, print_only: bool):
    """Launch a local multi-node network using quip-network-node (separate processes)."""
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
        for i in range(count):
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


# Entry points for console_scripts

def network_node_main():
    quip_network_node(standalone_mode=False)


def network_simulator_main():
    quip_network_simulator(standalone_mode=False)

