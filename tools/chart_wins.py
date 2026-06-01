#!/usr/bin/env python3
"""Render charts summarizing the validated QUIP PoW wins.

Consumes the artifacts written by ``tools/download_and_validate_wins.py``
(``<prefix>.wins.jsonl`` + ``<prefix>.validation.jsonl``) and produces a
multi-panel PNG telling the story of *who won, with what energy, how often*.

Usage::

    .quip/bin/python tools/chart_wins.py --in quip_wins --out quip_wins_charts.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(prefix: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load the wins archive and validation verdicts, sorted by block number."""
    wins = [json.loads(l) for l in Path(f"{prefix}.wins.jsonl").read_text().splitlines() if l.strip()]
    vals = [json.loads(l) for l in Path(f"{prefix}.validation.jsonl").read_text().splitlines() if l.strip()]
    wins.sort(key=lambda w: w["block_number"])
    vals.sort(key=lambda v: v["block_number"])
    return wins, vals


def concentration_metrics(win_counts: List[int]) -> Dict[str, float]:
    """Quantify how centralized mining is, given each miner's win count.

    ``win_counts`` is the list of per-miner win totals (one int per distinct
    miner, order irrelevant). Returns a dict of named metrics that the chart
    annotates onto the Pareto panel.

    DESIGN DECISION (your contribution — see the request below):
    There are several standard ways to express "how concentrated is this
    distribution", each with tradeoffs:
      - top-1 share / top-3 share: intuitive, but ignores the tail shape
      - HHI (Herfindahl-Hirschman Index): sum of squared shares, 1/n..1
      - Gini coefficient: 0 (perfectly even) .. ->1 (one miner takes all)
    Pick the metric(s) that best communicate decentralization for a PoW chain
    and return them keyed by a short label (used verbatim in the chart text).
    """
    total = sum(win_counts)
    n = len(win_counts)
    if total == 0 or n == 0:
        return {}
    desc = sorted(win_counts, reverse=True)
    shares = np.array(win_counts, dtype=float) / total
    hhi = float(np.sum(shares ** 2))
    # Gini via the ascending-sorted discrete formula: Σ(2i-n-1)·xᵢ / (n·Σxᵢ).
    asc = np.sort(np.array(win_counts, dtype=float))
    idx = np.arange(1, n + 1)
    gini = float(np.sum((2 * idx - n - 1) * asc) / (n * total))
    return {
        "top-1 share": f"{desc[0] / total:.0%}",
        "top-3 share": f"{sum(desc[:3]) / total:.0%}",
        "Gini": f"{gini:.2f}",
        "HHI": f"{hhi:.3f} (floor {1 / n:.3f})",
    }


def _panel_miner_pareto(ax: plt.Axes, wins: List[Dict[str, Any]]) -> None:
    """Bar chart of wins per miner (descending) with a cumulative-share line."""
    counts = Counter(w["miner"] for w in wins)
    ordered = counts.most_common()
    labels = [m[:10] for m, _ in ordered]
    values = [c for _, c in ordered]
    total = sum(values)
    cum = np.cumsum(values) / total * 100.0

    ax.bar(range(len(values)), values, color="#4c72b0")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("wins")
    ax.set_title(f"Wins per miner (n={total} wins, {len(values)} miners)")

    ax2 = ax.twinx()
    ax2.plot(range(len(values)), cum, color="#c44e52", marker=".", linewidth=1)
    ax2.set_ylabel("cumulative %", color="#c44e52")
    ax2.set_ylim(0, 105)

    metrics = concentration_metrics(values)
    txt = "\n".join(f"{k}: {v}" for k, v in metrics.items())
    ax.text(0.98, 0.55, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))


def _panel_energy_vs_block(ax: plt.Axes, wins: List[Dict[str, Any]]) -> None:
    """Winning energy vs block height, with the difficulty bar overlaid."""
    bn = [w["block_number"] for w in wins]
    e = [w["energy_milli"] / 1000.0 for w in wins]
    bar = [w["difficulty"]["max_energy_milli"] / 1000.0 for w in wins]
    ax.scatter(bn, e, s=12, color="#4c72b0", label="win energy")
    ax.plot(bn, bar, color="#c44e52", linewidth=1, label="difficulty bar (max energy)")
    ax.set_xlabel("block height")
    ax.set_ylabel("energy")
    ax.set_title("Winning energy vs block height")
    ax.legend(fontsize=7)


def _panel_margin_hist(ax: plt.Axes, wins: List[Dict[str, Any]]) -> None:
    """Histogram of how far below the bar each winner landed (lower = better)."""
    margin = [(w["difficulty"]["max_energy_milli"] - w["energy_milli"]) / 1000.0 for w in wins]
    ax.hist(margin, bins=40, color="#55a868")
    ax.set_yscale("log")  # early wins beat the bar by ~12000; recent by <100
    ax.set_xlabel("energy margin below bar (higher = beat it more decisively)")
    ax.set_ylabel("wins (log)")
    med = float(np.median(margin))
    ax.set_title(f"How decisively winners beat the bar (median {med:.0f})")


def _panel_cadence(ax: plt.Axes, wins: List[Dict[str, Any]]) -> None:
    """Block gap between consecutive wins over time (mining cadence)."""
    bn = [w["block_number"] for w in wins]
    gaps = np.diff(bn)
    ax.plot(bn[1:], gaps, color="#8172b3", marker=".", linewidth=0.8, markersize=3)
    ax.set_xlabel("block height")
    ax.set_ylabel("blocks since previous win")
    ax.set_title(f"Win cadence (median gap {int(np.median(gaps))} blocks)")


def _panel_diversity(ax: plt.Axes, vals: List[Dict[str, Any]]) -> None:
    """Diversity distribution against the required gate."""
    div = [v["diversity"] for v in vals if "diversity" in v]
    gate = next((v["threshold"]["min_diversity_milli"] / 1000.0
                 for v in vals if "threshold" in v), 0.2)
    ax.hist(div, bins=30, color="#ccb974")
    ax.axvline(gate, color="#c44e52", linestyle="--", label=f"gate {gate:.2f}")
    ax.set_xlabel("solution diversity")
    ax.set_ylabel("wins")
    ax.set_title("Diversity vs required gate")
    ax.legend(fontsize=7)


def _panel_top_miner_share(ax: plt.Axes, wins: List[Dict[str, Any]]) -> None:
    """Cumulative win count over block height for the top few miners."""
    counts = Counter(w["miner"] for w in wins)
    top = [m for m, _ in counts.most_common(5)]
    for miner in top:
        bn = [w["block_number"] for w in wins if w["miner"] == miner]
        ax.plot(bn, np.arange(1, len(bn) + 1), marker=".", markersize=3,
                linewidth=1, label=miner[:10])
    ax.set_xlabel("block height")
    ax.set_ylabel("cumulative wins")
    ax.set_title("Top-5 miners: cumulative wins over time")
    ax.legend(fontsize=7)


def build_figure(prefix: str, out_path: str) -> None:
    """Render the full multi-panel figure and write it to ``out_path``."""
    wins, vals = _load(prefix)
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    _panel_miner_pareto(axes[0][0], wins)
    _panel_energy_vs_block(axes[0][1], wins)
    _panel_margin_hist(axes[1][0], wins)
    _panel_cadence(axes[1][1], wins)
    _panel_diversity(axes[2][0], vals)
    _panel_top_miner_share(axes[2][1], wins)
    fig.suptitle(f"QUIP PoW wins — {len(wins)} validated blocks", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in", dest="prefix", default="quip_wins",
                   help="input artifact prefix (default: quip_wins)")
    p.add_argument("--out", default="quip_wins_charts.png",
                   help="output PNG path (default: quip_wins_charts.png)")
    args = p.parse_args()
    build_figure(args.prefix, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
