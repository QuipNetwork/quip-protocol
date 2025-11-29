#!/usr/bin/env python3
"""Click-based CLI for IPFS operations on Akash mining results.

Usage:
    python akash/ipfs_cli.py list                          # List deployments (uses cache)
    python akash/ipfs_cli.py list -r                       # List with forced refresh
    python akash/ipfs_cli.py list -v                       # List with individual CIDs
    python akash/ipfs_cli.py download <deployment_id>      # Download all instances
    python akash/ipfs_cli.py download <deployment_id> --latest  # Download latest only
    python akash/ipfs_cli.py download <cid>                # Download by CID
    python akash/ipfs_cli.py delete <deployment_id> --yes   # Delete by deployment ID
    python akash/ipfs_cli.py delete <cid> --yes            # Delete by CID

Cache: Manifests are cached in ~/.cache/quip-ipfs/ for 5 minutes.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cache settings
CACHE_DIR = Path.home() / ".cache" / "quip-ipfs"
CACHE_FILE = CACHE_DIR / "manifests.json"
CACHE_MAX_AGE = 300  # 5 minutes

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


def try_decompress(data: bytes) -> bytes:
    """Try to decompress gzipped data, return original if not gzipped."""
    try:
        return gzip.decompress(data)
    except (gzip.BadGzipFile, OSError):
        return data


def is_gzipped(data: bytes) -> bool:
    """Check if data is gzipped (magic bytes 1f 8b)."""
    return len(data) >= 2 and data[0] == 0x1f and data[1] == 0x8b


def is_manifest(data: bytes) -> bool:
    """Check if data is a valid gzipped manifest JSON."""
    # Only accept gzipped manifests (ignore old plaintext ones)
    if not is_gzipped(data):
        return False
    try:
        decompressed = gzip.decompress(data)
        obj = json.loads(decompressed)
        return "deployment_id" in obj and "results" in obj
    except (json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile, OSError):
        return False


def parse_manifest(data: bytes) -> Dict[str, Any]:
    """Parse manifest JSON (handles gzipped data)."""
    decompressed = try_decompress(data)
    return json.loads(decompressed)


def is_cid(value: str) -> bool:
    """Check if value looks like an IPFS CID (v0 or v1)."""
    return value.startswith("Qm") or value.startswith("bafy")


def load_cache(node: str) -> Optional[Dict[str, Any]]:
    """Load cached manifests if valid and not expired."""
    if not CACHE_FILE.exists():
        return None
    try:
        cache = json.loads(CACHE_FILE.read_text())
        # Check if cache is for same node and not expired
        if cache.get("node") != node:
            return None
        if time.time() - cache.get("timestamp", 0) > CACHE_MAX_AGE:
            return None
        return cache
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(node: str, manifests: List[Dict[str, Any]]) -> None:
    """Save manifests to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {
        "node": node,
        "timestamp": time.time(),
        "manifests": manifests,
    }
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def clear_cache() -> None:
    """Clear the manifest cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


def get_all_manifests(
    node: str,
    api_key: Optional[str],
    gateway: str,
    verbose: bool = False,
    refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch all pinned manifests from IPFS node (uses cache unless refresh=True)."""
    # Try cache first
    if not refresh:
        cache = load_cache(node)
        if cache:
            manifests = cache["manifests"]
            age = int(time.time() - cache["timestamp"])
            if verbose:
                click.echo(f"Using cached data ({age}s old, {len(manifests)} manifests)")
            return manifests

    if verbose:
        click.echo("Fetching manifests from IPFS node...")

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
    cids = list(keys.keys())

    if not cids:
        return []

    if verbose:
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

    # Save to cache
    save_cache(node, manifests)

    return manifests


def find_manifests_by_deployment(
    manifests: List[Dict[str, Any]],
    deployment_id: str,
) -> List[Dict[str, Any]]:
    """Filter manifests by deployment ID (exact match or prefix match)."""
    # Try exact match first
    exact = [m for m in manifests if m.get("deployment_id") == deployment_id]
    if exact:
        return exact

    # Try prefix match (for truncated IDs from list display)
    prefix = [m for m in manifests if m.get("deployment_id", "").startswith(deployment_id)]
    return prefix


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
@click.option("--verbose", "-v", is_flag=True, help="Show individual manifest CIDs")
@click.option("--refresh", "-r", is_flag=True, help="Force refresh from IPFS (ignore cache)")
@click.pass_context
def list_manifests(ctx: click.Context, verbose: bool, refresh: bool):
    """List all deployments on the IPFS node, grouped by deployment ID."""
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]
    gateway = ctx.obj["gateway"]

    click.echo(f"Connecting to IPFS node: {node}")

    manifests = get_all_manifests(node, api_key, gateway, verbose=verbose, refresh=refresh)

    if not manifests:
        click.echo("No manifest files found on IPFS node")
        return

    # Group manifests by deployment_id
    deployments: Dict[str, List[Dict[str, Any]]] = {}
    for m in manifests:
        dep_id = m.get("deployment_id", "unknown")
        if dep_id not in deployments:
            deployments[dep_id] = []
        deployments[dep_id].append(m)

    # Build summary for each deployment
    deployment_summaries = []
    for dep_id, dep_manifests in deployments.items():
        # Sort by timestamp to find latest
        sorted_manifests = sorted(
            dep_manifests,
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )
        latest = sorted_manifests[0]
        deployment_summaries.append({
            "deployment_id": dep_id,
            "latest_timestamp": latest.get("timestamp", "unknown"),
            "miner_type": latest.get("miner_type", "unknown"),
            "providers": len(dep_manifests),
            "manifests": sorted_manifests,
        })

    # Sort by latest timestamp (most recent first)
    deployment_summaries.sort(key=lambda x: x["latest_timestamp"], reverse=True)

    # Display grouped deployments
    click.echo()
    click.echo(f"{'Deployment ID':<25} {'Latest Timestamp':<20} {'Type':<8} {'Providers':<10}")
    click.echo("-" * 65)

    for dep in deployment_summaries:
        dep_id = dep["deployment_id"][:25]
        timestamp = dep["latest_timestamp"][:20]
        miner_type = dep["miner_type"][:8]
        providers = dep["providers"]
        click.echo(f"{dep_id:<25} {timestamp:<20} {miner_type:<8} {providers:<10}")

        if verbose:
            for m in dep["manifests"]:
                cid = m["_cid"]
                ts = m.get("timestamp", "unknown")[:20]
                click.echo(f"    {ts}  {cid}")

    click.echo()
    click.echo(f"Total deployments: {len(deployments)}, Total manifests: {len(manifests)}")


def download_manifest(
    manifest: Dict[str, Any],
    node: str,
    api_key: Optional[str],
    gateway: str,
    output_path: Path,
) -> Path:
    """Download a single manifest and its associated files.

    Returns the deployment directory path.
    """
    deployment_id = manifest.get("deployment_id", "unknown")
    miner_type = manifest.get("miner_type", "unknown")
    timestamp = manifest.get("timestamp", "unknown")
    results = manifest.get("results", {})
    cid = manifest.get("_cid", "unknown")

    click.echo(f"Deployment: {deployment_id} ({miner_type})")
    click.echo(f"Timestamp: {timestamp}")

    # Create deployment directory with timestamp to distinguish multiple instances
    deployment_dir = output_path / miner_type / deployment_id / timestamp
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest_file = deployment_dir / "manifest.json"
    manifest_json = json.dumps(manifest, indent=2)
    manifest_file.write_text(manifest_json)
    click.echo(f"  Saved: {manifest_file}")

    # Download JSON results
    json_cid = results.get("json_cid")
    if json_cid:
        click.echo(f"Downloading results JSON: {json_cid}")
        try:
            json_data = fetch_ipfs_content(json_cid, node, gateway, api_key)
            # Decompress if gzipped
            json_data = try_decompress(json_data)
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
            # Decompress if gzipped
            log_data = try_decompress(log_data)
            log_file = deployment_dir / "mining.log"
            log_file.write_bytes(log_data)
            click.echo(f"  Saved: {log_file}")
        except click.ClickException as e:
            click.echo(f"  Failed to download log: {e}", err=True)

    return deployment_dir


@cli.command()
@click.argument("identifier")
@click.option(
    "--output-dir",
    "-o",
    default="./ipfs_results",
    type=click.Path(),
    help="Output directory for downloaded files",
)
@click.option(
    "--latest",
    is_flag=True,
    help="Download only the latest manifest (when using deployment ID)",
)
@click.option(
    "--refresh",
    "-r",
    is_flag=True,
    help="Force refresh manifest list from IPFS (ignore cache)",
)
@click.pass_context
def download(ctx: click.Context, identifier: str, output_dir: str, latest: bool, refresh: bool):
    """Download deployment results from IPFS.

    IDENTIFIER can be either:
    - A manifest CID (starts with Qm or bafy)
    - A deployment ID (downloads all instances, or latest with --latest)
    """
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]
    gateway = ctx.obj["gateway"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if is_cid(identifier):
        # Direct CID download (original behavior)
        click.echo(f"Fetching CID: {identifier}")

        try:
            data = fetch_ipfs_content(identifier, node, gateway, api_key)
        except click.ClickException as e:
            raise click.ClickException(f"Failed to fetch manifest: {e}")

        if not is_manifest(data):
            raise click.ClickException(
                f"CID {identifier} does not appear to be a valid manifest. "
                "Expected JSON with 'deployment_id' and 'results' fields."
            )

        manifest = parse_manifest(data)
        manifest["_cid"] = identifier
        deployment_dir = download_manifest(manifest, node, api_key, gateway, output_path)
        click.echo()
        click.echo(f"Download complete: {deployment_dir}")

    else:
        # Deployment ID - search for matching manifests
        click.echo(f"Searching for deployment: {identifier}")
        click.echo(f"Connecting to IPFS node: {node}")

        all_manifests = get_all_manifests(node, api_key, gateway, refresh=refresh)
        matching = find_manifests_by_deployment(all_manifests, identifier)

        if not matching:
            raise click.ClickException(
                f"No manifests found for deployment ID: {identifier}\n"
                "Use 'list' command to see available deployments."
            )

        # Sort by timestamp (most recent first)
        matching.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        if latest:
            # Download only the latest
            click.echo(f"Found {len(matching)} manifest(s), downloading latest...")
            click.echo()
            deployment_dir = download_manifest(
                matching[0], node, api_key, gateway, output_path
            )
            click.echo()
            click.echo(f"Download complete: {deployment_dir}")
        else:
            # Download all
            click.echo(f"Found {len(matching)} manifest(s), downloading all...")
            click.echo()
            dirs = []
            for i, manifest in enumerate(matching, 1):
                click.echo(f"--- Manifest {i}/{len(matching)} ---")
                deployment_dir = download_manifest(
                    manifest, node, api_key, gateway, output_path
                )
                dirs.append(deployment_dir)
                click.echo()

            click.echo(f"Download complete: {len(dirs)} manifest(s) saved to {output_path}")


def unpin_cid(node: str, api_key: str, cid: str) -> bool:
    """Unpin a single CID. Returns True on success."""
    try:
        response = ipfs_api_call(node, f"/api/v0/pin/rm?arg={cid}", api_key)
        result = json.loads(response)
        pins_removed = result.get("Pins", [])
        return cid in pins_removed
    except click.ClickException:
        return False


@cli.command()
@click.argument("identifier")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--gc", is_flag=True, help="Run garbage collection after unpinning")
@click.option("--refresh", "-r", is_flag=True, help="Force refresh manifest list from IPFS")
@click.pass_context
def delete(ctx: click.Context, identifier: str, yes: bool, gc: bool, refresh: bool):
    """Delete (unpin) deployment data from the IPFS node.

    IDENTIFIER can be either:
    - A manifest CID (starts with Qm or bafy)
    - A deployment ID (deletes all manifests and their results)

    This removes the pins but does not immediately delete the data.
    Use --gc to run garbage collection after unpinning.
    """
    node = ctx.obj["node"]
    api_key = ctx.obj["api_key"]
    gateway = ctx.obj["gateway"]

    if not api_key:
        raise click.ClickException(
            "API key required for delete operations. "
            "Set IPFS_API_KEY or use --api-key option."
        )

    # Collect all CIDs to delete
    cids_to_delete: List[str] = []

    if is_cid(identifier):
        # Single CID deletion
        cids_to_delete.append(identifier)
        click.echo(f"Will delete CID: {identifier}")
    else:
        # Deployment ID - find all related manifests and their CIDs
        click.echo(f"Searching for deployment: {identifier}")
        all_manifests = get_all_manifests(node, api_key, gateway, refresh=refresh)
        matching = find_manifests_by_deployment(all_manifests, identifier)

        if not matching:
            raise click.ClickException(
                f"No manifests found for deployment ID: {identifier}\n"
                "Use 'list' command to see available deployments."
            )

        # Collect manifest CIDs and their associated result CIDs
        for manifest in matching:
            manifest_cid = manifest.get("_cid")
            if manifest_cid:
                cids_to_delete.append(manifest_cid)
            results = manifest.get("results", {})
            json_cid = results.get("json_cid")
            log_cid = results.get("log_cid")
            if json_cid:
                cids_to_delete.append(json_cid)
            if log_cid:
                cids_to_delete.append(log_cid)

        click.echo(f"Found {len(matching)} manifest(s) with {len(cids_to_delete)} total CIDs")
        for m in matching:
            click.echo(f"  - {m.get('deployment_id')} @ {m.get('timestamp')}")

    # Confirm deletion
    if not yes:
        click.echo(f"\nAbout to unpin {len(cids_to_delete)} CID(s) from: {node}")
        if not click.confirm("Are you sure?"):
            click.echo("Aborted.")
            return

    # Unpin all CIDs
    click.echo(f"\nUnpinning {len(cids_to_delete)} CID(s)...")
    success_count = 0
    fail_count = 0
    for cid in cids_to_delete:
        if unpin_cid(node, api_key, cid):
            click.echo(f"  Unpinned: {cid}")
            success_count += 1
        else:
            click.echo(f"  Failed: {cid}", err=True)
            fail_count += 1

    click.echo(f"\nUnpinned {success_count}/{len(cids_to_delete)} CIDs")
    if fail_count > 0:
        click.echo(f"Warning: {fail_count} CID(s) failed to unpin", err=True)

    # Clear the cache since we modified pins
    clear_cache()

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
