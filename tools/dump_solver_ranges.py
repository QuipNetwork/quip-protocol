#!/usr/bin/env python3
"""Dump D-Wave QPU solver h/J ranges to markdown + JSON.

Enumerates every QPU solver the configured API key can access, extracts the
range and qubit-count fields from `solver.properties`, and writes:

  - docs/dwave-solver-ranges.md   (human reference; hand-written sections are
                                   preserved across regeneration via HTML
                                   comment markers)
  - docs/dwave-solver-ranges.json (machine-readable sibling)

Always prints a markdown table to stdout.

Usage:
    python tools/dump_solver_ranges.py
    python tools/dump_solver_ranges.py --regions na-west-1
    python tools/dump_solver_ranges.py --format json
    python tools/dump_solver_ranges.py --stdout-only
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from dwave.cloud import Client  # noqa: E402  (must follow load_dotenv)

DWAVE_REGIONS = ["na-west-1", "na-east-1", "eu-central-1"]
BEGIN_MARKER = "<!-- BEGIN GENERATED -->"
END_MARKER = "<!-- END GENERATED -->"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "docs"
MARKDOWN_NAME = "dwave-solver-ranges.md"
JSON_NAME = "dwave-solver-ranges.json"


@dataclass
class SolverRanges:
    """Range and capacity facts for a single D-Wave QPU chip."""

    chip_id: str
    solver_names: list[str] = field(default_factory=list)
    topology_type: Optional[str] = None
    topology_shape: Optional[list[int]] = None
    num_qubits: Optional[int] = None
    num_active_qubits: Optional[int] = None
    h_range: Optional[list[float]] = None
    j_range: Optional[list[float]] = None
    extended_j_range: Optional[list[float]] = None
    per_qubit_coupling_range: Optional[list[float]] = None
    regions: list[str] = field(default_factory=list)
    online: bool = True


def _coerce_range(value: Any) -> Optional[list[float]]:
    """Return a 2-element float list, or None if the property is absent/malformed."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _extract(solver: Any, region: str) -> Optional[SolverRanges]:
    """Build a SolverRanges from one solver, or None if it's not a QPU."""
    props = solver.properties or {}
    if props.get("category") != "qpu":
        return None
    topology = props.get("topology") or {}
    num_qubits = props.get("num_qubits")
    qubits = props.get("qubits") or []
    return SolverRanges(
        chip_id=str(props.get("chip_id") or solver.name),
        solver_names=[solver.name],
        topology_type=topology.get("type"),
        topology_shape=list(topology.get("shape")) if topology.get("shape") else None,
        num_qubits=int(num_qubits) if num_qubits is not None else None,
        num_active_qubits=len(qubits) if qubits else None,
        h_range=_coerce_range(props.get("h_range")),
        j_range=_coerce_range(props.get("j_range")),
        extended_j_range=_coerce_range(props.get("extended_j_range")),
        per_qubit_coupling_range=_coerce_range(props.get("per_qubit_coupling_range")),
        regions=[region],
        online=bool(getattr(solver, "status", None) is None or solver.status == "ONLINE"),
    )


def _merge(existing: SolverRanges, addition: SolverRanges) -> None:
    """Merge a duplicate chip seen in another region into the existing entry."""
    for name in addition.solver_names:
        if name not in existing.solver_names:
            existing.solver_names.append(name)
    for region in addition.regions:
        if region not in existing.regions:
            existing.regions.append(region)


def discover(regions: list[str], include_offline: bool) -> list[SolverRanges]:
    """Enumerate QPU solvers across regions and return de-duplicated range facts.

    Args:
        regions: D-Wave SAPI region slugs to query.
        include_offline: If False, drop solvers whose status is not online.

    Returns:
        List of SolverRanges sorted by topology type then chip_id.
    """
    by_chip: dict[str, SolverRanges] = {}
    for region in regions:
        try:
            with Client.from_config(region=region) as client:
                solvers = client.get_solvers()
        except Exception as exc:  # noqa: BLE001 - SAPI raises a wide variety
            print(f"warning: could not query region {region}: {exc}", file=sys.stderr)
            continue
        for solver in solvers:
            entry = _extract(solver, region)
            if entry is None:
                continue
            if not include_offline and not entry.online:
                continue
            if entry.chip_id in by_chip:
                _merge(by_chip[entry.chip_id], entry)
            else:
                by_chip[entry.chip_id] = entry
    return sorted(by_chip.values(), key=lambda s: (s.topology_type or "", s.chip_id))


def _fmt_range(value: Optional[list[float]]) -> str:
    """Render a range pair for the markdown table; missing values become n/a."""
    if value is None:
        return "n/a"
    lo, hi = value
    return f"[{lo:g}, {hi:g}]"


def _fmt_qubits(active: Optional[int], total: Optional[int]) -> str:
    if active is None and total is None:
        return "n/a"
    return f"{active if active is not None else '?'}/{total if total is not None else '?'}"


def _fmt_topology(topology_type: Optional[str], shape: Optional[list[int]]) -> str:
    if not topology_type:
        return "n/a"
    if shape:
        return f"{topology_type} {tuple(shape)}"
    return topology_type


def format_markdown_table(solvers: list[SolverRanges]) -> str:
    """Render the per-solver markdown table (no headers, no surrounding prose)."""
    headers = [
        "Chip ID",
        "Topology",
        "Qubits (active/total)",
        "h_range",
        "j_range",
        "extended_j_range",
        "per_qubit_coupling_range",
        "Regions",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    if not solvers:
        lines.append("| _(no QPU solvers accessible)_ |" + "|".join([" "] * (len(headers) - 1)) + "|")
        return "\n".join(lines)
    for s in solvers:
        suffix = "" if s.online else " _(offline)_"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s.chip_id}`{suffix}",
                    _fmt_topology(s.topology_type, s.topology_shape),
                    _fmt_qubits(s.num_active_qubits, s.num_qubits),
                    _fmt_range(s.h_range),
                    _fmt_range(s.j_range),
                    _fmt_range(s.extended_j_range),
                    _fmt_range(s.per_qubit_coupling_range),
                    ", ".join(s.regions) or "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_generated_block(solvers: list[SolverRanges], cmd: str) -> str:
    """Render the auto-generated region between BEGIN/END markers."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"{BEGIN_MARKER}\n"
        f"_Last generated: {timestamp}. Regenerate with `{cmd}`._\n\n"
        f"{format_markdown_table(solvers)}\n"
        f"{END_MARKER}"
    )


def update_markdown_file(path: Path, generated: str) -> None:
    """Rewrite only the BEGIN/END marker region; preserve everything else.

    Creates the file with a skeleton if it doesn't yet exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create the doc skeleton first (committed alongside this tool)."
        )
    text = path.read_text(encoding="utf-8")
    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if begin == -1 or end == -1 or end < begin:
        raise ValueError(
            f"{path} is missing {BEGIN_MARKER}/{END_MARKER} markers; refusing to overwrite."
        )
    end_complete = end + len(END_MARKER)
    new_text = text[:begin] + generated + text[end_complete:]
    path.write_text(new_text, encoding="utf-8")


def build_json_doc(solvers: list[SolverRanges], cmd: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": cmd,
        "solvers": [asdict(s) for s in solvers],
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regions",
        default=",".join(DWAVE_REGIONS),
        help=f"Comma-separated D-Wave regions (default: {','.join(DWAVE_REGIONS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        default="md,json",
        help="Comma-separated output formats: md, json (default: md,json)",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print to stdout; do not write any files.",
    )
    parser.add_argument(
        "--include-offline",
        action="store_true",
        help="Include solvers reported as not online.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    formats = {f.strip().lower() for f in args.format.split(",") if f.strip()}
    unknown = formats - {"md", "json"}
    if unknown:
        print(f"error: unknown format(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    solvers = discover(regions, args.include_offline)
    if not solvers:
        print(
            "error: no accessible QPU solvers. Check DWAVE_API_KEY in .env and region access.",
            file=sys.stderr,
        )
        return 1

    cmd = "python tools/dump_solver_ranges.py"
    print(format_markdown_table(solvers))

    if args.stdout_only:
        return 0

    if "md" in formats:
        update_markdown_file(args.output_dir / MARKDOWN_NAME, build_generated_block(solvers, cmd))
        print(f"wrote {args.output_dir / MARKDOWN_NAME}", file=sys.stderr)
    if "json" in formats:
        json_path = args.output_dir / JSON_NAME
        json_path.write_text(
            json.dumps(build_json_doc(solvers, cmd), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
