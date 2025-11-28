#!/usr/bin/env python3
"""Click-based CLI for IPFS operations on Akash mining results.

Usage:
    python akash/ipfs_cli.py list
    python akash/ipfs_cli.py download <cid>
    python akash/ipfs_cli.py delete <cid> --yes
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv

# Load .env file - check akash dir first, then project root
_script_dir = Path(__file__).parent
_env_paths = [
    _script_dir / ".env",
    _script_dir.parent / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break


def ipfs_api_call(
    node: str,
    endpoint: str,
    api_key: Optional[str] = None,
    method: str = "POST",
    timeout: int = 30,
) -> bytes:
    """Make an authenticated API call to the IPFS node."""
    url = f"{node.rstrip('/')}{endpoint}"
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise click.ClickException(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise click.ClickException(f"Connection failed: {e.reason}")


def fetch_ipfs_content(
    cid: str,
    node: str,
    gateway: str,
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> bytes:
    """Fetch content from IPFS, trying local node first then public gateway."""
    # Try local node first
    try:
        url = f"{node.rstrip('/')}/ipfs/{cid}"
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass

    # Fallback to public gateway
    try:
        url = f"{gateway.rstrip('/')}/ipfs/{cid}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise click.ClickException(f"Failed to fetch CID {cid}: {e}")


def is_manifest(data: bytes) -> bool:
    """Check if data is a valid manifest JSON."""
    try:
        obj = json.loads(data)
        return "deployment_id" in obj and "results" in obj
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def parse_manifest(data: bytes) -> Dict[str, Any]:
    """Parse manifest JSON."""
    return json.loads(data)


@click.group()
@click.option(
    "--node",
    envvar="IPFS_NODE",
    default="https://carback-ipfs.ngrok.io",
    help="IPFS node URL",
)
@click.option(
    "--api-key",
    envvar="IPFS_API_KEY",
    default=None,
    help="API key for IPFS node authentication",
)
@click.option(
    "--gateway",
    envvar="IPFS_GATEWAY",
    default="https://ipfs.io",
    help="Public IPFS gateway for fallback",
)
@click.pass_context
def cli(ctx: click.Context, node: str, api_key: Optional[str], gateway: str):
    """IPFS CLI for managing Akash mining results.

    Environment variables:
      IPFS_NODE     - IPFS node URL
      IPFS_API_KEY  - API key for authentication
      IPFS_GATEWAY  - Public gateway fallback
    """
    ctx.ensure_object(dict)
    ctx.obj["node"] = node
    ctx.obj["api_key"] = api_key
    ctx.obj["gateway"] = gateway


@cli.command(name="list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def list_manifests(ctx: click.Context, verbose: bool):
    """List all manifests/deployments on the IPFS node."""
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]
    gateway = ctx.obj["gateway"]

    click.echo(f"Connecting to IPFS node: {node}")

    # List all pins
    try:
        response = ipfs_api_call(node, "/api/v0/pin/ls?type=recursive", api_key)
        pins_data = json.loads(response)
    except click.ClickException:
        raise
    except json.JSONDecodeError:
        raise click.ClickException("Invalid JSON response from IPFS node")

    # Extract CIDs from response {"Keys": {"Qm...": {"Type": "recursive"}, ...}}
    keys = pins_data.get("Keys", {})
    cids = [k for k in keys.keys()]

    if not cids:
        click.echo("No pinned files found on IPFS node")
        return

    click.echo(f"Found {len(cids)} pinned CID(s), scanning for manifests...")

    # Identify manifests
    manifests: List[Dict[str, Any]] = []
    for cid in cids:
        try:
            data = fetch_ipfs_content(cid, node, gateway, api_key, timeout=30)
            if is_manifest(data):
                manifest = parse_manifest(data)
                manifest["_cid"] = cid
                manifests.append(manifest)
                if verbose:
                    click.echo(f"  Found manifest: {cid}")
        except click.ClickException:
            if verbose:
                click.echo(f"  Skipping {cid} (not a manifest or fetch failed)")

    if not manifests:
        click.echo("No manifest files found among pinned CIDs")
        return

    # Display manifests
    click.echo()
    click.echo(f"{'Deployment ID':<20} {'Type':<8} {'Timestamp':<20} {'Manifest CID':<50}")
    click.echo("-" * 100)

    for m in sorted(manifests, key=lambda x: x.get("timestamp", "")):
        deployment_id = m.get("deployment_id", "unknown")[:20]
        miner_type = m.get("miner_type", "unknown")[:8]
        timestamp = m.get("timestamp", "unknown")[:20]
        cid = m["_cid"]
        click.echo(f"{deployment_id:<20} {miner_type:<8} {timestamp:<20} {cid:<50}")

    click.echo()
    click.echo(f"Total manifests: {len(manifests)}")


@cli.command()
@click.argument("cid")
@click.option(
    "--output-dir",
    "-o",
    default="./ipfs_results",
    type=click.Path(),
    help="Output directory for downloaded files",
)
@click.pass_context
def download(ctx: click.Context, cid: str, output_dir: str):
    """Download deployment results from IPFS.

    CID can be a manifest CID or will attempt to fetch and parse as manifest.
    """
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]
    gateway = ctx.obj["gateway"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Fetching CID: {cid}")

    # Fetch the manifest
    try:
        data = fetch_ipfs_content(cid, node, gateway, api_key)
    except click.ClickException as e:
        raise click.ClickException(f"Failed to fetch manifest: {e}")

    if not is_manifest(data):
        raise click.ClickException(
            f"CID {cid} does not appear to be a valid manifest. "
            "Expected JSON with 'deployment_id' and 'results' fields."
        )

    manifest = parse_manifest(data)
    deployment_id = manifest.get("deployment_id", "unknown")
    miner_type = manifest.get("miner_type", "unknown")
    timestamp = manifest.get("timestamp", "unknown")
    results = manifest.get("results", {})

    click.echo(f"Deployment: {deployment_id} ({miner_type})")
    click.echo(f"Timestamp: {timestamp}")

    # Create deployment directory
    deployment_dir = output_path / miner_type / deployment_id
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest_file = deployment_dir / "manifest.json"
    manifest_file.write_bytes(data)
    click.echo(f"  Saved: {manifest_file}")

    # Download JSON results
    json_cid = results.get("json_cid")
    if json_cid:
        click.echo(f"Downloading results JSON: {json_cid}")
        try:
            json_data = fetch_ipfs_content(json_cid, node, gateway, api_key)
            json_file = deployment_dir / "results.json"
            json_file.write_bytes(json_data)
            click.echo(f"  Saved: {json_file}")

            # Extract and display key metrics
            try:
                results_obj = json.loads(json_data)
                blocks = results_obj.get("results", {}).get("blocks_found", "N/A")
                rate = results_obj.get("results", {}).get("blocks_per_minute", "N/A")
                click.echo(f"  Results: {blocks} blocks, {rate} blocks/min")
            except json.JSONDecodeError:
                pass
        except click.ClickException as e:
            click.echo(f"  Failed to download JSON: {e}", err=True)

    # Download log file
    log_cid = results.get("log_cid")
    if log_cid:
        click.echo(f"Downloading log: {log_cid}")
        try:
            log_data = fetch_ipfs_content(log_cid, node, gateway, api_key)
            log_file = deployment_dir / "mining.log"
            log_file.write_bytes(log_data)
            click.echo(f"  Saved: {log_file}")
        except click.ClickException as e:
            click.echo(f"  Failed to download log: {e}", err=True)

    click.echo()
    click.echo(f"Download complete: {deployment_dir}")


@cli.command()
@click.argument("cid")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--gc", is_flag=True, help="Run garbage collection after unpinning")
@click.pass_context
def delete(ctx: click.Context, cid: str, yes: bool, gc: bool):
    """Delete (unpin) a CID from the IPFS node.

    This removes the pin but does not immediately delete the data.
    Use --gc to run garbage collection after unpinning.
    """
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]

    if not api_key:
        raise click.ClickException(
            "API key required for delete operations. "
            "Set IPFS_API_KEY or use --api-key option."
        )

    # Confirm deletion
    if not yes:
        click.echo(f"About to unpin CID: {cid}")
        click.echo(f"From node: {node}")
        if not click.confirm("Are you sure?"):
            click.echo("Aborted.")
            return

    # Unpin the CID
    click.echo(f"Unpinning CID: {cid}")
    try:
        response = ipfs_api_call(node, f"/api/v0/pin/rm?arg={cid}", api_key)
        result = json.loads(response)
        pins_removed = result.get("Pins", [])
        if cid in pins_removed:
            click.echo(f"Successfully unpinned: {cid}")
        else:
            click.echo(f"Unpin response: {result}")
    except click.ClickException as e:
        raise click.ClickException(f"Failed to unpin: {e}")

    # Optionally run garbage collection
    if gc:
        click.echo("Running garbage collection...")
        try:
            response = ipfs_api_call(node, "/api/v0/repo/gc", api_key, timeout=120)
            click.echo("Garbage collection complete")
        except click.ClickException as e:
            click.echo(f"Warning: GC failed: {e}", err=True)


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """Check connection to the IPFS node."""
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]

    click.echo(f"IPFS Node: {node}")
    click.echo(f"API Key: {'configured' if api_key else 'not set'}")

    # Try to get node ID/version
    try:
        response = ipfs_api_call(node, "/api/v0/id", api_key, timeout=10)
        info = json.loads(response)
        click.echo(f"Node ID: {info.get('ID', 'unknown')}")
        click.echo(f"Agent: {info.get('AgentVersion', 'unknown')}")
        click.echo("Status: Connected")
    except click.ClickException as e:
        click.echo(f"Status: Connection failed - {e}", err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
