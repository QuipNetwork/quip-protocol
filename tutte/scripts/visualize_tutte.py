#!/usr/bin/env python
"""
Tutte Engine Visualizer — Flask + vis-network with live SSE streaming.

Layout:
  Row 1 — Input Graph + Contributing Graphs (side by side)
  Row 2 — Result
  Row 3 — Summary
  Row 4 — Timeline (live-streamed via SSE)

Usage:
    python scripts/visualize_tutte.py
    Then open http://localhost:5002/?atlas=18

URL Parameters:
    atlas=N          — NetworkX graph atlas index (0–1252)
    dwave_topo=zephyr&dwave_m=1&dwave_t=1 — D-Wave topology (zephyr/pegasus/chimera)
    edges=0-1,1-2,2-0 — Custom edge list
    rand_n=12&rand_m=18 — Random graph with n nodes and m edges
    timeout=60       — Engine timeout in seconds (default 60)
    threshold=100    — Timeline bottleneck threshold in ms (default 100)
    engine=synthesis — Engine: "synthesis", "algebraic", or "hybrid"
"""

import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from flask import Flask, Response, request
from tutte.graph import Graph
from tutte.logs import EventType, LogLevel, get_log, reset_log
from tutte.lookup.core import load_default_table
from tutte.synthesis.base import SynthesisResult
from tutte.synthesis.engine import SynthesisEngine

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poly_to_html(poly_str: str) -> str:
    """Convert polynomial string to readable HTML with superscripts."""
    html = re.sub(r'\*\*(\d+)', r'<sup>\1</sup>', str(poly_str))
    html = html.replace('*', '')
    return html


def factored_poly_html(tutte_poly) -> str:
    """Try to factor a TuttePolynomial via SymPy. Returns HTML string.

    Shows factored form if non-trivial, otherwise falls back to expanded form.
    """
    from sympy import factor, symbols

    x, y = symbols('x y')
    expr = sum(coeff * x**i * y**j for (i, j), coeff in tutte_poly._coeffs.items())
    factored = factor(expr)

    # If factored form is shorter, use it; otherwise show expanded
    fact_str = str(factored)
    exp_str = str(expr)
    display = fact_str if len(fact_str) < len(exp_str) else exp_str

    return poly_to_html(display)


def _normalize_nx_graph(G):
    """Return G with nodes relabeled to consecutive ints starting at 0.

    Several networkx generators (grid_2d_graph, kneser_graph, balanced_tree
    with certain inputs) use tuple or frozenset node labels that vis.js
    can't render and that break spring_layout keying in the visualizer.
    Relabeling to ints keeps every family path uniform.
    """
    if G is None:
        return G
    return nx.convert_node_labels_to_integers(G)


def parse_graph(args) -> tuple:
    """Parse URL parameters into a (nx.Graph, description) tuple.

    The returned graph always has integer node labels (0..n-1) so
    downstream rendering and engine conversion paths don't need to
    special-case tuple/frozenset labels from networkx generators.
    """
    G, desc = _parse_graph_raw(args)
    return _normalize_nx_graph(G), desc


def _parse_graph_raw(args) -> tuple:
    """Inner: returns the raw nx.Graph from the appropriate generator
    without normalizing node labels."""
    atlas = args.get("atlas", type=int)
    if atlas is not None:
        try:
            G = nx.graph_atlas(atlas)
            return G, f"Atlas #{atlas}"
        except Exception as e:
            return None, f"Invalid atlas index: {e}"

    dwave_topo = args.get("dwave_topo", "").strip()
    dwave_m = args.get("dwave_m", type=int)
    dwave_t = args.get("dwave_t", type=int)
    if dwave_topo and dwave_m is not None:
        try:
            import dwave_networkx as dnx
            if dwave_topo == "zephyr":
                t = dwave_t if dwave_t is not None else 1
                G = dnx.zephyr_graph(dwave_m, t)
                return G, f"Zephyr Z({dwave_m},{t})"
            elif dwave_topo == "pegasus":
                if dwave_m < 2:
                    return None, "Pegasus requires m >= 2 (P(1) is empty)"
                G = dnx.pegasus_graph(dwave_m)
                return G, f"Pegasus P({dwave_m})"
            elif dwave_topo == "chimera":
                # D-Wave Chimera is specified by a single parameter m (tile grid
                # is m×m, shore size is fixed at 4 on every D-Wave processor).
                G = dnx.chimera_graph(dwave_m)
                return G, f"Chimera C({dwave_m})"
            else:
                return None, f"Unknown D-Wave topology: {dwave_topo}"
        except ImportError:
            return None, "dwave-networkx not installed"
        except Exception as e:
            return None, f"Invalid D-Wave params: {e}"

    edges_str = args.get("edges", "").strip()
    if edges_str:
        try:
            G = nx.Graph()
            for part in edges_str.split(","):
                u, v = part.strip().split("-")
                G.add_edge(int(u), int(v))
            return G, f"Custom ({G.number_of_edges()} edges)"
        except Exception as e:
            return None, f"Invalid edge list: {e}"

    # Random graph: rand_n=12&rand_m=12
    rand_n = args.get("rand_n", type=int)
    rand_m = args.get("rand_m", type=int)
    if rand_n is not None and rand_m is not None:
        max_edges = rand_n * (rand_n - 1) // 2
        if rand_m > max_edges:
            return None, f"Too many edges: {rand_n} nodes can have at most {max_edges} edges"
        if rand_n < 1:
            return None, "Need at least 1 node"
        G = nx.gnm_random_graph(rand_n, rand_m)
        return G, f"Random G({rand_n},{rand_m}) — {G.number_of_nodes()}n, {G.number_of_edges()}e"

    # Graph family: family=complete&n=5 or family=grid&n=3&m=4
    family = args.get("family", "").strip()
    if family:
        n = args.get("n", 5, type=int)
        m = args.get("m", 0, type=int)
        try:
            G, desc = _build_family_graph(family, n, m)
            return G, desc
        except Exception as e:
            return None, f"Invalid family params: {e}"

    return None, ""


# Map of family name → (generator, needs_m, label_fn)
GRAPH_FAMILIES = {
    "complete": ("Complete K_n", False),
    "cycle": ("Cycle C_n", False),
    "path": ("Path P_n", False),
    "wheel": ("Wheel W_n", False),
    "star": ("Star S_n", False),
    "complete_bipartite": ("Complete Bipartite K_{n,m}", True),
    "grid": ("Grid G_{n,m}", True),
    "ladder": ("Ladder L_n", False),
    "petersen": ("Petersen", False),
    "tutte": ("Tutte", False),
    "dodecahedral": ("Dodecahedral", False),
    "icosahedral": ("Icosahedral", False),
    "octahedral": ("Octahedral", False),
    "cubical": ("Cubical", False),
    "tetrahedral": ("Tetrahedral", False),
    "heawood": ("Heawood", False),
    "moebius_kantor": ("Moebius-Kantor", False),
    "bull": ("Bull", False),
    "chvatal": ("Chvatal", False),
    "desargues": ("Desargues", False),
    "pappus": ("Pappus", False),
    "gear": ("Gear G_n", False),
    "prism": ("Prism P_n", False),
    "friendship": ("Friendship F_n", False),
    "barbell": ("Barbell B_{n,m}", True),
    "empty": ("Empty E_n", False),
    "random_tree": ("Random Tree T_n", False),
    "balanced_tree": ("Balanced Tree B_{r,h}", True),
    "kneser": ("Kneser K_{n,k}", True),
    "k_regular": ("k-Regular R_{k,n}", True),
}


def _build_family_graph(family: str, n: int, m: int):
    """Build a named graph family. Returns (nx.Graph, description)."""
    if family == "complete":
        return nx.complete_graph(n), f"K_{n}"
    elif family == "cycle":
        return nx.cycle_graph(n), f"C_{n}"
    elif family == "path":
        return nx.path_graph(n), f"P_{n}"
    elif family == "wheel":
        return nx.wheel_graph(n), f"W_{n}"
    elif family == "star":
        return nx.star_graph(n), f"S_{n}"
    elif family == "complete_bipartite":
        return nx.complete_bipartite_graph(n, m or n), f"K_{{{n},{m or n}}}"
    elif family == "grid":
        return nx.grid_2d_graph(n, m or n), f"Grid({n},{m or n})"
    elif family == "ladder":
        return nx.ladder_graph(n), f"Ladder({n})"
    elif family == "petersen":
        return nx.petersen_graph(), "Petersen"
    elif family == "tutte":
        return nx.tutte_graph(), "Tutte"
    elif family == "dodecahedral":
        return nx.dodecahedral_graph(), "Dodecahedral"
    elif family == "icosahedral":
        return nx.icosahedral_graph(), "Icosahedral"
    elif family == "octahedral":
        return nx.octahedral_graph(), "Octahedral"
    elif family == "cubical":
        return nx.cubical_graph(), "Cubical"
    elif family == "tetrahedral":
        return nx.tetrahedral_graph(), "Tetrahedral"
    elif family == "heawood":
        return nx.heawood_graph(), "Heawood"
    elif family == "moebius_kantor":
        return nx.moebius_kantor_graph(), "Moebius-Kantor"
    elif family == "bull":
        return nx.bull_graph(), "Bull"
    elif family == "chvatal":
        return nx.chvatal_graph(), "Chvatal"
    elif family == "desargues":
        return nx.desargues_graph(), "Desargues"
    elif family == "pappus":
        return nx.pappus_graph(), "Pappus"
    elif family == "gear":
        G = nx.wheel_graph(n)
        # Insert vertex on each spoke
        gear = nx.Graph()
        hub = 0
        outer = list(range(1, n))
        for i, v in enumerate(outer):
            gear.add_edge(hub, n + i)
            gear.add_edge(n + i, v)
            gear.add_edge(v, outer[(i + 1) % len(outer)])
        return gear, f"Gear({n})"
    elif family == "prism":
        return nx.circular_ladder_graph(n), f"Prism({n})"
    elif family == "friendship":
        G = nx.Graph()
        for i in range(n):
            G.add_edges_from([(0, 2*i+1), (0, 2*i+2), (2*i+1, 2*i+2)])
        return G, f"Friendship({n})"
    elif family == "barbell":
        return nx.barbell_graph(n, m), f"Barbell({n},{m})"
    elif family == "empty":
        return nx.empty_graph(n), f"E_{n}"
    elif family == "random_tree":
        return nx.random_labeled_tree(n), f"RandomTree({n})"
    elif family == "balanced_tree":
        r = n  # branching factor
        h = m if m else 2  # height
        return nx.balanced_tree(r, h), f"BalancedTree({r},{h})"
    elif family == "kneser":
        if m >= n:
            raise ValueError(f"Kneser K(n,k) requires k < n, got n={n}, k={m}")
        return nx.kneser_graph(n, m or 1), f"Kneser({n},{m or 1})"
    elif family == "k_regular":
        k = n  # degree
        num_nodes = m if m else 10  # number of nodes
        if k >= num_nodes:
            raise ValueError(f"k-Regular requires k < n, got k={k}, n={num_nodes}")
        if k * num_nodes % 2 != 0:
            raise ValueError(f"k-Regular requires k*n even, got k={k}, n={num_nodes}")
        return nx.random_regular_graph(k, num_nodes), f"Regular({k},{num_nodes})"
    else:
        raise ValueError(f"Unknown family: {family}")


def vis_data_json(G) -> tuple:
    """Convert nx.Graph to vis-network JSON (nodes, edges)."""
    pos = nx.spring_layout(G, seed=42, scale=250)
    nodes = []
    for n in G.nodes():
        nodes.append({
            "id": n, "label": str(n),
            "x": pos[n][0], "y": pos[n][1],
            "physics": False,
        })
    edges = []
    for u, v in G.edges():
        edges.append({"from": u, "to": v})
    return json.dumps(nodes), json.dumps(edges)


def small_graph_vis(G, div_id) -> str:
    """Return vis-network JS snippet for a small graph panel."""
    nodes_json, edges_json = vis_data_json(G)
    return (
        f"(function(){{"
        f"var n=new vis.Network("
        f"document.getElementById('{div_id}'),"
        f"{{nodes:new vis.DataSet({nodes_json}),edges:new vis.DataSet({edges_json})}},"
        f"opts);"
        f"n.fit({{padding:20}});"
        f"}})();"
    )


def _synth_graph_to_nx(g_obj):
    """Convert a tutte Graph or MultiGraph synthesized during a run to an nx.Graph.

    Used by the visualizer to render engine-synthesized sub-problems that
    are not in the rainbow table. MultiGraphs are flattened to simple
    networkx graphs; parallel edges and loops are dropped in this view
    (node positions and connectivity are the visually useful parts).
    """
    if g_obj is None:
        return None
    if hasattr(g_obj, 'to_networkx'):
        try:
            return g_obj.to_networkx()
        except Exception:
            pass
    edge_counts = getattr(g_obj, 'edge_counts', None)
    nodes = getattr(g_obj, 'nodes', None)
    if edge_counts is not None and nodes is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edge_counts.keys())
        return G
    return None


def graph_from_entry(entry):
    """Reconstruct nx.Graph from a rainbow table MinorEntry."""
    if entry.graph is not None:
        return entry.graph.to_networkx()

    from tutte.graphs.covering import _minor_to_graph
    g = _minor_to_graph(entry)
    if g is not None:
        return g.to_networkx()

    if entry.name.startswith("atlas_"):
        try:
            idx = int(entry.name[6:])
            return nx.graph_atlas(idx)
        except (ValueError, nx.NetworkXError):
            pass

    return None


# ---------------------------------------------------------------------------
# Event colors (shared between Python summary & JS)
# ---------------------------------------------------------------------------

EVENT_COLORS = {
    "cache_hit": "#2e7d32",
    "cache_miss": "#9e9e9e",
    "lookup_hit": "#2e7d32",
    "lookup_miss": "#9e9e9e",
    "base_case": "#2e7d32",
    "factorize": "#1565c0",
    "series_parallel": "#6a1b9a",
    "treewidth_dp": "#00838f",
    "ksum": "#1565c0",
    "family_recognition": "#2e7d32",
    "vf2_match": "#e65100",
    "tile_accept": "#1565c0",
    "cover_result": "#1565c0",
    "edge_add": "#c62828",
    "multigraph_op": "#9e9e9e",
    "verify": "#2e7d32",
    "theorem6": "#e65100",
    "hierarchical": "#e65100",
    "synthesis_start": "#9e9e9e",
    "candidate_filter": "#9e9e9e",
}


# ---------------------------------------------------------------------------
# HTML template — page loads immediately, timeline streams via SSE
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tutte Engine Visualizer</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; margin: 0; padding: 16px; background: #fafafa; }}
    h2 {{ font-size: 14px; margin: 0 0 8px 0; color: #333; }}
    .graphs-row {{ display: flex; gap: 16px; margin-bottom: 16px; }}
    .graphs-row > .panel {{ flex: 1; display: flex; flex-direction: column; }}
    .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .graph-box {{ height: 320px; border: 1px solid #eee; border-radius: 4px; }}
    .small-graph {{ height: 140px; border: 1px solid #eee; border-radius: 4px; }}
    .meta {{ font-size: 11px; color: #555; line-height: 1.6; }}
    .meta b {{ color: #333; }}
    .result-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 12px; }}
    .result-grid dt {{ color: #888; }}
    .result-grid dd {{ margin: 0; }}
    .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; color: #fff; font-size: 10px; font-weight: bold; }}
    .timeline-scroll {{ max-height: 400px; overflow-y: auto; }}
    .timeline {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
    .timeline th {{ background: #f5f5f5; position: sticky; top: 0; text-align: left; padding: 4px 6px; }}
    .timeline td {{ padding: 2px 6px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
    .timeline td:last-child {{ white-space: normal; word-break: break-word; }}
    .summary {{ border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    .summary th {{ background: #f5f5f5; text-align: left; padding: 5px 10px; border-bottom: 2px solid #ddd; white-space: nowrap; }}
    .summary th:nth-child(2), .summary th:nth-child(3), .summary th:nth-child(4) {{ text-align: right; }}
    .summary td {{ padding: 4px 10px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
    .summary tr:hover {{ background: #fafafa; }}
    .summary .pct-bar {{ display: inline-block; height: 6px; border-radius: 3px; vertical-align: middle; }}
    .minors-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .minor-card {{ flex: 1; min-width: 200px; border: 1px solid #eee; border-radius: 4px; padding: 6px; }}
    .minor-label {{ font-size: 11px; font-weight: bold; margin-bottom: 4px; }}
    input[type=number] {{ font-family: inherit; padding: 3px 6px; width: 60px; border: 1px solid #ccc; border-radius: 3px; }}
    input[type=text] {{ font-family: inherit; padding: 3px 6px; width: 100%; border: 1px solid #ccc; border-radius: 3px; }}
    select {{ font-family: inherit; padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; }}
    button {{ font-family: inherit; padding: 3px 10px; cursor: pointer; border: 1px solid #ccc; border-radius: 3px; background: #fff; }}
    button:hover {{ background: #f0f0f0; }}
    .controls {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 12px; }}
    .poly {{ font-size: 12px; line-height: 1.5; word-break: break-word; }}
    .timeout-banner {{ background: #fff3e0; border: 1px solid #e65100; border-radius: 4px; padding: 8px; color: #e65100; font-weight: bold; }}
    .error-banner {{ background: #fce4ec; border: 1px solid #c62828; border-radius: 4px; padding: 8px; color: #c62828; }}
    .section {{ margin-bottom: 16px; }}
    .ctrl-grid {{ display: flex; gap: 32px; }}
    .ctrl-group {{ display: flex; flex-direction: column; gap: 6px; }}
    .ctrl-label {{ font-size: 11px; font-weight: bold; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
    .ctrl-radio {{ font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px; }}
    .ctrl-radio input[type="radio"] {{ margin: 0; }}
    .ctrl-indent {{ margin-left: 20px; font-size: 12px; }}
    .ctrl-row {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
    .run-btn {{ background: #1565c0; color: #fff; border: none; padding: 6px 20px; border-radius: 4px; font-weight: bold; font-size: 12px; cursor: pointer; }}
    .run-btn:hover {{ background: #0d47a1; }}
    input:disabled, select:disabled {{ opacity: 0.4; }}
    .spinner {{ display: inline-block; width: 12px; height: 12px; border: 2px solid #ccc; border-top-color: #333; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 6px; vertical-align: middle; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div class="panel section" style="padding:16px;">
    <h2 style="margin-bottom:12px;">Control Panel</h2>
    <form method="get" action="/" id="ctrl-form">
      <input type="hidden" name="form_submitted" value="1">
      <div class="ctrl-grid">
        <!-- Input source selection -->
        <div class="ctrl-group">
          <div class="ctrl-label">Input Graph</div>
          <label class="ctrl-radio"><input type="radio" name="source" value="atlas" {atlas_checked}> Atlas index</label>
          <div class="ctrl-indent"><input type="number" name="atlas" value="{atlas_val}" placeholder="#" min="0" max="1252" style="width:80px" {atlas_disabled}></div>
          <label class="ctrl-radio"><input type="radio" name="source" value="dwave" {dwave_checked}> D-Wave topology</label>
          <div class="ctrl-indent">
            <select name="dwave_topo" id="dwave-topo-select" {dwave_disabled}>{dwave_topo_options}</select>
            <span id="dwave-m-wrap"><span id="dwave-m-label">{dwave_m_label}</span>=<input type="number" name="dwave_m" value="{dwave_m_val}" placeholder="1" min="1" style="width:50px" {dwave_disabled}></span>
            <span id="dwave-t-wrap" style="{dwave_t_display}"><span id="dwave-t-label">{dwave_t_label}</span>=<input type="number" name="dwave_t" value="{dwave_t_val}" placeholder="1" min="1" style="width:50px" {dwave_disabled}></span>
          </div>
          <label class="ctrl-radio"><input type="radio" name="source" value="family" {family_checked}> Graph family</label>
          <div class="ctrl-indent">
            <select name="family" id="family-select" {family_disabled}>{family_options}</select>
            <span id="n-wrap"><span id="n-label">{n_label}</span>=<input type="number" name="n" value="{n_val}" placeholder="5" min="1" style="width:50px" {family_disabled}></span>
            <span id="m-wrap" style="display:none"><span id="m-label">{m_label}</span>=<input type="number" name="m" value="{m_val}" placeholder="" min="0" style="width:50px" {family_disabled}></span>
          </div>
          <label class="ctrl-radio"><input type="radio" name="source" value="edges" {edges_checked}> Edge list</label>
          <div class="ctrl-indent"><input type="text" name="edges" value="{edges_val}" placeholder="0-1,1-2,2-3,3-0" style="width:260px" {edges_disabled}></div>
          <label class="ctrl-radio"><input type="radio" name="source" value="random" {random_checked}> Random graph</label>
          <div class="ctrl-indent">
            nodes=<input type="number" name="rand_n" value="{rand_n_val}" placeholder="12" min="1" max="200" style="width:55px" {random_disabled}>
            edges=<input type="number" name="rand_m" value="{rand_m_val}" placeholder="12" min="0" style="width:55px" {random_disabled}>
            <span id="rand-max-edges" style="color:#999;font-size:11px">{rand_max_hint}</span>
          </div>
        </div>
        <!-- Settings -->
        <div class="ctrl-group">
          <div class="ctrl-label">Settings</div>
          <div class="ctrl-row">
            <span>Engine:</span>
            <select name="engine">{engine_options}</select>
          </div>
          <div class="ctrl-row">
            <span>Timeout:</span>
            <input type="number" name="timeout" value="{timeout_val}" min="1" max="3600" style="width:60px">s
          </div>
          <div class="ctrl-row">
            <label><input type="checkbox" name="debug" value="1" {debug_checked}> Debug logging</label>
          </div>
          <div class="ctrl-row">
            <label title="Unchecking skips the direct rainbow-table lookup for the target graph only. Sub-problems still consult the table."><input type="checkbox" name="use_lookup" value="1" {use_lookup_checked}> Lookup target in rainbow table</label>
          </div>
          <div class="ctrl-row" style="margin-top:12px;">
            <button type="submit" class="run-btn" style="background:#555;">Load Graph</button>
            <button type="button" class="run-btn" id="run-engine-btn" onclick="startEngine()">Run Engine</button>
            <button type="button" class="run-btn" id="stop-engine-btn" style="background:#c62828;display:none;" onclick="stopEngine()">Stop</button>
          </div>
        </div>
      </div>
    </form>
  </div>
  <script>
    // Enable/disable inputs based on radio selection
    (function() {{
      var form = document.getElementById('ctrl-form');
      var radios = form.querySelectorAll('input[name="source"]');
      function update() {{
        var sel = form.querySelector('input[name="source"]:checked').value;
        form.querySelector('input[name="atlas"]').disabled = (sel !== 'atlas');
        form.querySelector('select[name="dwave_topo"]').disabled = (sel !== 'dwave');
        form.querySelector('input[name="dwave_m"]').disabled = (sel !== 'dwave');
        form.querySelector('input[name="dwave_t"]').disabled = (sel !== 'dwave');
        form.querySelector('select[name="family"]').disabled = (sel !== 'family');
        form.querySelector('input[name="n"]').disabled = (sel !== 'family');
        form.querySelector('input[name="m"]').disabled = (sel !== 'family');
        form.querySelector('input[name="edges"]').disabled = (sel !== 'edges');
        form.querySelector('input[name="rand_n"]').disabled = (sel !== 'random');
        form.querySelector('input[name="rand_m"]').disabled = (sel !== 'random');
        if (sel === 'dwave') updateDwave();
        if (sel === 'random') updateRandMax();
      }}
      radios.forEach(function(r) {{ r.addEventListener('change', update); }});
      update();

      // Show/hide n and m based on selected family
      var needsM = {{'complete_bipartite':1,'grid':1,'barbell':1,'balanced_tree':1,'kneser':1,'k_regular':1}};
      var fixed = {{'petersen':1,'tutte':1,'dodecahedral':1,'icosahedral':1,'octahedral':1,
        'cubical':1,'tetrahedral':1,'heawood':1,'moebius_kantor':1,'bull':1,
        'chvatal':1,'desargues':1,'pappus':1}};
      var paramLabels = {{
        'complete': ['nodes', ''],
        'cycle': ['nodes', ''],
        'path': ['nodes', ''],
        'wheel': ['nodes (incl. hub)', ''],
        'star': ['leaves', ''],
        'complete_bipartite': ['partition 1', 'partition 2'],
        'grid': ['rows', 'columns'],
        'ladder': ['rungs', ''],
        'gear': ['spokes', ''],
        'prism': ['sides', ''],
        'friendship': ['triangles', ''],
        'barbell': ['clique size', 'path length'],
        'empty': ['nodes', ''],
        'random_tree': ['nodes', ''],
        'balanced_tree': ['branching factor', 'height'],
        'kneser': ['n', 'k'],
        'k_regular': ['degree k', 'nodes'],
      }};
      var fsel = document.getElementById('family-select');
      function updateFamily() {{
        var v = fsel.value;
        document.getElementById('m-wrap').style.display = needsM[v] ? '' : 'none';
        document.getElementById('n-wrap').style.display = fixed[v] ? 'none' : '';
        var labels = paramLabels[v] || ['n', 'm'];
        document.getElementById('n-label').textContent = labels[0];
        document.getElementById('m-label').textContent = labels[1] || 'm';
      }}
      fsel.addEventListener('change', updateFamily);
      updateFamily();

      // Show/hide D-Wave params based on topology
      var dsel = document.getElementById('dwave-topo-select');
      var dwaveLabels = {{
        'zephyr':  ['grid parameter', 'tile parameter'],
        'pegasus': ['size parameter', ''],
        'chimera': ['grid parameter', ''],
      }};
      function updateDwave() {{
        var topo = dsel.value;
        // Zephyr needs (m, t). Pegasus and Chimera both take just m.
        var showT = (topo === 'zephyr');
        document.getElementById('dwave-t-wrap').style.display = showT ? '' : 'none';
        var labels = dwaveLabels[topo] || ['m', 't'];
        document.getElementById('dwave-m-label').textContent = labels[0];
        document.getElementById('dwave-t-label').textContent = labels[1] || 't';
        var mInput = form.querySelector('input[name="dwave_m"]');
        if (topo === 'pegasus') {{
          mInput.min = 2;
          if (parseInt(mInput.value) < 2) mInput.value = 2;
        }} else {{
          mInput.min = 1;
        }}
      }}
      dsel.addEventListener('change', updateDwave);
      updateDwave();

      // Random graph: show max edges hint
      var randN = form.querySelector('input[name="rand_n"]');
      var randM = form.querySelector('input[name="rand_m"]');
      var randHint = document.getElementById('rand-max-edges');
      function updateRandMax() {{
        var n = parseInt(randN.value) || 0;
        var maxE = n * (n - 1) / 2;
        randHint.textContent = n > 0 ? '(max ' + maxE + ')' : '';
        if (parseInt(randM.value) > maxE && maxE > 0) randM.value = maxE;
        randM.max = maxE > 0 ? maxE : '';
      }}
      randN.addEventListener('input', updateRandMax);
      updateRandMax();
    }})();
  </script>

  <!-- Row 1: Input Graph + Contributing Graphs side by side -->
  <div class="graphs-row">
    <div class="panel">
      <h2>Input Graph — {graph_desc}</h2>
      <div class="meta">{input_meta}</div>
      <div id="input-graph" class="graph-box"></div>
    </div>
    <div class="panel">
      <h2>Contributing Graphs</h2>
      <div id="minors-container" class="minors-grid">
        <div class="meta">Click "Run Engine" to start.</div>
      </div>
    </div>
  </div>

  <!-- Row 2: Result -->
  <div class="panel section">
    <h2>Result</h2>
    <div id="result-container">
      <div class="meta">Click "Run Engine" to start.</div>
    </div>
  </div>

  <!-- Row 3: Summary -->
  <div class="panel section">
    <h2>Summary</h2>
    <div id="summary-container"></div>
  </div>

  <!-- Row 4: Timeline -->
  <div class="panel section">
    <h2>Timeline <span id="event-count"></span>
      <span id="engine-elapsed" style="font-weight:normal;color:#888;font-size:12px;margin-left:12px;"></span>
    </h2>
    <div class="timeline-scroll" id="timeline-scroll">
      <table class="timeline" id="timeline-table">
        <tr><th>Time</th><th>Duration</th><th>D</th><th>Type</th><th>Module</th><th>Message</th><th>Graph</th></tr>
      </table>
    </div>
  </div>

  <!-- Row 5: Step-graph viewer -->
  <div class="panel section">
    <h2>Step Graph <span id="step-label" style="font-weight:normal;color:#888;font-size:12px;"></span></h2>
    <div class="controls" id="step-controls">
      <button type="button" id="step-first">&#x21E4; First</button>
      <button type="button" id="step-prev">Prev</button>
      <button type="button" id="step-play">Play</button>
      <button type="button" id="step-next">Next</button>
      <button type="button" id="step-last">Last &#x21E5;</button>
      <span style="color:#666;margin-left:8px">interval</span>
      <input type="number" id="step-interval" value="500" min="50" max="10000" step="50" style="width:70px">ms
      <span id="step-graph-count" style="color:#888;margin-left:12px;"></span>
    </div>
    <div id="step-graph" class="graph-box"></div>
  </div>

  <script>
    var EVENT_COLORS = {event_colors_json};
    var opts = {{ edges: {{ smooth: false }}, physics: {{ enabled: false }} }};
    var _es = null;  // current EventSource
    var _engineTimerId = null;
    var _engineStartedAt = 0;

    function _fmtElapsed(ms) {{
      var s = ms / 1000;
      if (s < 60) return s.toFixed(1) + 's';
      var m = Math.floor(s / 60);
      var r = (s - m * 60).toFixed(1);
      return m + 'm ' + r + 's';
    }}

    function _paintElapsed() {{
      var el = _fmtElapsed(performance.now() - _engineStartedAt);
      var h = document.getElementById('engine-elapsed');
      if (h) h.textContent = 'elapsed ' + el;
      var r = document.getElementById('running-elapsed');
      if (r) r.textContent = el;
    }}

    function _startEngineClock() {{
      _engineStartedAt = performance.now();
      _paintElapsed();
      if (_engineTimerId) clearInterval(_engineTimerId);
      _engineTimerId = setInterval(_paintElapsed, 250);
    }}

    function _stopEngineClock() {{
      if (_engineTimerId) {{ clearInterval(_engineTimerId); _engineTimerId = null; }}
      _paintElapsed();
    }}

    window.addEventListener('DOMContentLoaded', function() {{
      // Render input graph
      {input_graph_script}
    }});

    function stopEngine() {{
      if (_es) {{ _es.close(); _es = null; }}
      _stopEngineClock();
      document.getElementById('run-engine-btn').style.display = '';
      document.getElementById('stop-engine-btn').style.display = 'none';
    }}

    function startEngine() {{
      // Close any previous connection
      if (_es) {{ _es.close(); _es = null; }}

      // Show stop button, hide run button
      document.getElementById('run-engine-btn').style.display = 'none';
      document.getElementById('stop-engine-btn').style.display = '';

      // Reset UI
      var tbody = document.getElementById('timeline-table');
      while (tbody.rows.length > 1) tbody.deleteRow(1);
      document.getElementById('event-count').textContent = '';
      document.getElementById('result-container').innerHTML =
        '<div class="meta"><span class="spinner"></span>Running engine... (<span id="running-elapsed">0.0s</span>)</div>';
      document.getElementById('minors-container').innerHTML = '<div class="meta"><span class="spinner"></span>Waiting for engine...</div>';
      document.getElementById('summary-container').innerHTML = '';
      // Reset step-graph viewer
      document.getElementById('step-graph').innerHTML = '';
      document.getElementById('step-label').textContent = '';
      document.getElementById('step-graph-count').textContent = '';

      _startEngineClock();
      {sse_script}
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

# Global state for the current engine run (single-user visualizer)
_engine_lock = threading.Lock()
_engine_state = {
    "running": False,
    "result": None,
    "error": None,
    "timed_out": False,
    "elapsed": 0.0,
    "done": False,
}


def _run_engine_thread(graph, table, timeout_sec, engine_type, skip_target_lookup=False):
    """Run the engine in a thread, storing result in _engine_state.

    If skip_target_lookup is True, the top-level synthesize() call skips the
    rainbow-table lookup for the input graph — but the engine still uses the
    full table for sub-problems encountered during decomposition.
    """
    global _engine_state
    result_holder = [None]
    error_holder = [None]

    def target():
        try:
            if engine_type == "hybrid":
                from tutte.synthesis.hybrid import HybridSynthesisEngine
                engine = HybridSynthesisEngine(table=table)
                engine.skip_target_lookup = skip_target_lookup
                hybrid_result = engine.synthesize(graph)
                result_holder[0] = SynthesisResult(
                    polynomial=hybrid_result.polynomial,
                    recipe=hybrid_result.recipe,
                    verified=hybrid_result.verified,
                    method=hybrid_result.method,
                    minors_used=getattr(hybrid_result, 'minors_used', set()),
                    synthesized_minors=getattr(hybrid_result, 'synthesized_minors', set()),
                    synthesized_graphs=getattr(hybrid_result, 'synthesized_graphs', {}),
                )
            elif engine_type == "algebraic":
                from tutte.synthesis.algebraic import AlgebraicSynthesisEngine
                engine = AlgebraicSynthesisEngine(table=table)
                alg_result = engine.synthesize(graph)
                result_holder[0] = SynthesisResult(
                    polynomial=alg_result.polynomial,
                    recipe=alg_result.recipe,
                    verified=alg_result.verified,
                    method=alg_result.method,
                )
            else:
                engine = SynthesisEngine(table=table)
                engine.skip_target_lookup = skip_target_lookup
                result_holder[0] = engine.synthesize(graph)
        except Exception as e:
            error_holder[0] = str(e)

    t0 = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    elapsed = time.perf_counter() - t0

    with _engine_lock:
        _engine_state["elapsed"] = elapsed
        if thread.is_alive():
            _engine_state["timed_out"] = True
        elif error_holder[0]:
            _engine_state["error"] = error_holder[0]
        else:
            _engine_state["result"] = result_holder[0]
        _engine_state["done"] = True


@app.route("/stream")
def stream():
    """SSE endpoint: streams events as they're recorded, then final result."""
    timeout_sec = request.args.get("timeout", 60, type=int)
    engine_type = request.args.get("engine", "synthesis")
    threshold_ms = request.args.get("threshold", 100, type=float)
    debug = request.args.get("debug", "0") == "1"
    # form_submitted sentinel distinguishes "no form submit" from "unchecked"
    form_submitted = request.args.get("form_submitted", "0") == "1"
    if form_submitted:
        use_lookup = request.args.get("use_lookup", "0") == "1"
    else:
        use_lookup = True

    G_nx, graph_desc = parse_graph(request.args)
    if G_nx is None:
        def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': graph_desc or 'No graph'})}\n\n"
        return Response(error_stream(), mimetype="text/event-stream")

    graph = Graph.from_networkx(G_nx)
    # Always load the full table; `skip_target_lookup` only gates the
    # top-level lookup for the input graph, leaving sub-problem lookups on.
    table = load_default_table()
    skip_target_lookup = not use_lookup

    # Reset log, set min_level, enable graph snapshot capture for the viewer.
    reset_log()
    log_ref = get_log()
    log_ref.min_level = LogLevel.DEBUG if debug else LogLevel.INFO
    log_ref.capture_graphs = True
    global _engine_state
    with _engine_lock:
        _engine_state = {
            "running": True, "result": None, "error": None,
            "timed_out": False, "elapsed": 0.0, "done": False,
        }

    # Start engine in background thread
    engine_thread = threading.Thread(
        target=_run_engine_thread,
        args=(graph, table, timeout_sec, engine_type, skip_target_lookup),
        daemon=True,
    )
    engine_thread.start()

    def _serialize_batch(new_events, start_idx, prev_timestamp):
        """Serialize a list of new events into a single SSE batch message.

        Returns (batch, last_timestamp). Each event dict may include a
        `graph_key` field referencing a snapshot sent in the same batch.
        """
        batch = []
        for i, ev in enumerate(new_events):
            idx = start_idx + i
            gap = ev.timestamp - prev_timestamp if prev_timestamp is not None else 0.0
            color = EVENT_COLORS.get(ev.event_type.value, "#9e9e9e")
            batch.append({
                "index": idx,
                "timestamp": f"{ev.timestamp:.3f}s",
                "ts_raw": ev.timestamp,
                "depth": ev.depth,
                "event_type": ev.event_type.value,
                "module": ev.module,
                "message": ev.message,
                "color": color,
                "gap": gap,
                "graph_key": ev.graph_key,
            })
            prev_timestamp = ev.timestamp
        return batch, prev_timestamp

    def event_stream():
        log = get_log()
        sent_count = 0
        prev_timestamp = None
        poll_interval = 0.05  # 50ms
        sent_snapshot_keys: set = set()

        def _batch_payload(events, start_idx, prev_ts):
            batch, new_prev_ts = _serialize_batch(events, start_idx, prev_ts)
            new_snapshots = log.new_graph_snapshots(sent_snapshot_keys)
            if new_snapshots:
                sent_snapshot_keys.update(new_snapshots.keys())
            payload = {'type': 'batch', 'events': batch}
            if new_snapshots:
                payload['snapshots'] = new_snapshots
            return payload, new_prev_ts

        while True:
            new_events = log.events_since(sent_count)

            if new_events:
                payload, prev_timestamp = _batch_payload(
                    new_events, sent_count, prev_timestamp
                )
                sent_count += len(new_events)
                yield f"data: {json.dumps(payload)}\n\n"

            # Check if engine is done
            with _engine_lock:
                done = _engine_state["done"]

            if done:
                # Send any remaining events
                remaining = log.events_since(sent_count)
                if remaining:
                    payload, prev_timestamp = _batch_payload(
                        remaining, sent_count, prev_timestamp
                    )
                    sent_count += len(remaining)
                    yield f"data: {json.dumps(payload)}\n\n"

                # Build final result payload
                with _engine_lock:
                    result = _engine_state["result"]
                    error = _engine_state["error"]
                    timed_out = _engine_state["timed_out"]
                    elapsed = _engine_state["elapsed"]

                final = {
                    "type": "done",
                    "timed_out": timed_out,
                    "elapsed": elapsed,
                    "event_count": sent_count,
                    "threshold_ms": threshold_ms,
                }

                if timed_out:
                    final["result_html"] = (
                        f'<div class="timeout-banner">'
                        f'TIMEOUT after {elapsed:.1f}s (limit: {timeout_sec}s)<br>'
                        f'Check timeline for last event before timeout.'
                        f'</div>'
                    )
                    final["minors"] = []
                    final["minors_lookup"] = []
                    final["minors_synthesized"] = []
                elif error:
                    final["result_html"] = (
                        f'<div class="error-banner">Engine error: {error}</div>'
                    )
                    final["minors"] = []
                    final["minors_lookup"] = []
                    final["minors_synthesized"] = []
                elif result is not None:
                    poly_html = factored_poly_html(result.polynomial)
                    t11 = result.polynomial.num_spanning_trees()
                    verified_str = (
                        '<span style="color:#2e7d32">YES</span>' if result.verified
                        else '<span style="color:#c62828">NO</span>'
                    )
                    final["result_html"] = (
                        f'<dl class="result-grid">'
                        f'<dt>Method</dt><dd>{result.method}</dd>'
                        f'<dt>Verified</dt><dd>{verified_str}</dd>'
                        f'<dt>Tiles</dt><dd>{result.tiles_used}</dd>'
                        f'<dt>T(1,1)</dt><dd>{t11}</dd>'
                        f'<dt>Time</dt><dd>{elapsed:.3f}s</dd>'
                        f'<dt>Polynomial</dt><dd class="poly">{poly_html}</dd>'
                        f'</dl>'
                    )
                    # Build contributing graphs split by provenance:
                    #  - minors_lookup: entries the engine actually pulled from the rainbow table.
                    #  - minors_synthesized: sub-graphs the engine synthesized from scratch
                    #    during this run (may or may not also exist in the table).
                    lookup_list = []
                    for key in sorted(result.minors_used or []):
                        entry = table.get_entry_by_key(key)
                        if entry is None:
                            continue
                        card = {"name": entry.name, "edges": entry.edge_count}
                        minor_nx = graph_from_entry(entry)
                        if minor_nx is not None:
                            nodes_json, edges_json = vis_data_json(minor_nx)
                            card["nodes"] = nodes_json
                            card["edges_data"] = edges_json
                        lookup_list.append(card)
                        if len(lookup_list) >= 12:
                            break

                    synth_list = []
                    synthesized_graphs = getattr(result, 'synthesized_graphs', {}) or {}
                    synthesized_minors = getattr(result, 'synthesized_minors', set()) or set()
                    # Skip ones that came from the lookup table — those appear in lookup_list.
                    for key in sorted(synthesized_minors - (result.minors_used or set())):
                        g_obj = synthesized_graphs.get(key)
                        if g_obj is None:
                            continue
                        minor_nx = _synth_graph_to_nx(g_obj)
                        nc = getattr(g_obj, 'node_count', lambda: '?')()
                        ec = getattr(g_obj, 'edge_count', lambda: '?')()
                        name = (
                            f"{type(g_obj).__name__} {nc}n {ec}e "
                            f"[{key[:8]}]"
                        )
                        card = {"name": name, "edges": ec}
                        if minor_nx is not None:
                            nodes_json, edges_json = vis_data_json(minor_nx)
                            card["nodes"] = nodes_json
                            card["edges_data"] = edges_json
                        synth_list.append(card)
                        if len(synth_list) >= 24:
                            break

                    final["minors"] = lookup_list  # back-compat: legacy key
                    final["minors_lookup"] = lookup_list
                    final["minors_synthesized"] = synth_list
                else:
                    final["result_html"] = (
                        '<div class="error-banner">Engine returned no result.</div>'
                    )
                    final["minors"] = []
                    final["minors_lookup"] = []
                    final["minors_synthesized"] = []

                # Build summary from log aggregation
                summary_data = log.summary()
                total_time = prev_timestamp if prev_timestamp is not None else 0.0
                if summary_data and total_time > 0:
                    sorted_items = sorted(
                        ((et.value, (c, d)) for et, (c, d) in summary_data.items()),
                        key=lambda x: -x[1][1],
                    )
                    summary_rows = []
                    for etype, (count, total_dur) in sorted_items:
                        pct = (total_dur / total_time) * 100 if total_time > 0 else 0
                        color = EVENT_COLORS.get(etype, "#9e9e9e")
                        summary_rows.append({
                            "event_type": etype, "color": color,
                            "count": count, "duration": f"{total_dur:.3f}s",
                            "pct": f"{pct:.1f}%", "pct_num": round(pct, 1),
                        })
                    final["summary"] = summary_rows
                else:
                    final["summary"] = []

                yield f"data: {json.dumps(final)}\n\n"
                return

            time.sleep(poll_interval)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    timeout_sec = request.args.get("timeout", 60, type=int)
    threshold_ms = request.args.get("threshold", 100, type=float)
    engine_type = request.args.get("engine", "synthesis")
    debug = request.args.get("debug", "0") == "1"
    # use_lookup defaults to ON. Unchecking the form checkbox submits without
    # the param, so we detect form submission via a hidden sentinel.
    form_submitted = request.args.get("form_submitted", "0") == "1"
    if form_submitted:
        use_lookup = request.args.get("use_lookup", "0") == "1"
    else:
        use_lookup = True

    atlas_val = request.args.get("atlas", "")
    dwave_topo_val = request.args.get("dwave_topo", "zephyr")
    dwave_m_val = request.args.get("dwave_m", "1")
    dwave_t_val = request.args.get("dwave_t", "1")
    edges_val = request.args.get("edges", "")
    family_val = request.args.get("family", "")
    n_val = request.args.get("n", "5")
    m_val = request.args.get("m", "")

    rand_n_val = request.args.get("rand_n", "12")
    rand_m_val = request.args.get("rand_m", "12")

    # Determine which source is active
    source = request.args.get("source", "")
    if not source:
        if atlas_val:
            source = "atlas"
        elif request.args.get("dwave_topo"):
            source = "dwave"
        elif family_val:
            source = "family"
        elif edges_val:
            source = "edges"
        elif request.args.get("rand_n"):
            source = "random"
        else:
            source = "atlas"

    atlas_checked = "checked" if source == "atlas" else ""
    dwave_checked = "checked" if source == "dwave" else ""
    family_checked = "checked" if source == "family" else ""
    edges_checked = "checked" if source == "edges" else ""
    random_checked = "checked" if source == "random" else ""
    atlas_disabled = "" if source == "atlas" else "disabled"
    dwave_disabled = "" if source == "dwave" else "disabled"
    family_disabled = "" if source == "family" else "disabled"
    edges_disabled = "" if source == "edges" else "disabled"
    random_disabled = "" if source == "random" else "disabled"

    engine_options = ""
    for opt in ["synthesis", "hybrid", "algebraic"]:
        sel = " selected" if opt == engine_type else ""
        engine_options += f'<option value="{opt}"{sel}>{opt}</option>'

    # Build family dropdown options
    family_options = '<option value="">-- select --</option>'
    for key, (label, _needs_m) in GRAPH_FAMILIES.items():
        sel = " selected" if key == family_val else ""
        family_options += f'<option value="{key}"{sel}>{label}</option>'

    # Build D-Wave topology dropdown
    dwave_topo_options = ""
    for topo, label in [("zephyr", "Zephyr Z(m, t)"), ("pegasus", "Pegasus P(m)"), ("chimera", "Chimera C(m)")]:
        sel = " selected" if topo == dwave_topo_val else ""
        dwave_topo_options += f'<option value="{topo}"{sel}>{label}</option>'

    G_nx, graph_desc = parse_graph(request.args)

    debug_checked = "checked" if debug else ""
    use_lookup_checked = "checked" if use_lookup else ""

    # Compute server-side labels for D-Wave params
    _dwave_labels = {
        "zephyr": ("grid parameter", "tile parameter"),
        "pegasus": ("size parameter", ""),
        "chimera": ("grid parameter", ""),
    }
    dwave_m_label, dwave_t_label = _dwave_labels.get(dwave_topo_val, ("m", "t"))
    # Only Zephyr exposes a tile parameter; Pegasus and Chimera hide the t input.
    dwave_t_display = "" if dwave_topo_val == "zephyr" else "display:none"
    if not dwave_t_label:
        dwave_t_label = "t"

    # Compute server-side labels for family params
    _family_labels = {
        "complete": ("nodes", ""),
        "cycle": ("nodes", ""),
        "path": ("nodes", ""),
        "wheel": ("nodes (incl. hub)", ""),
        "star": ("leaves", ""),
        "complete_bipartite": ("partition 1", "partition 2"),
        "grid": ("rows", "columns"),
        "ladder": ("rungs", ""),
        "gear": ("spokes", ""),
        "prism": ("sides", ""),
        "friendship": ("triangles", ""),
        "barbell": ("clique size", "path length"),
        "empty": ("nodes", ""),
        "random_tree": ("nodes", ""),
        "balanced_tree": ("branching factor", "height"),
        "kneser": ("n", "k"),
        "k_regular": ("degree k", "nodes"),
    }
    n_label, m_label = _family_labels.get(family_val, ("n", "m"))
    if not m_label:
        m_label = "m"

    # Compute random graph max edges hint
    try:
        _rn = int(rand_n_val)
        rand_max_hint = f"(max {_rn * (_rn - 1) // 2})" if _rn > 0 else ""
    except (ValueError, TypeError):
        rand_max_hint = ""

    ctrl_vars = dict(
        atlas_val=atlas_val, dwave_m_val=dwave_m_val, dwave_t_val=dwave_t_val,
        dwave_topo_options=dwave_topo_options,
        dwave_m_label=dwave_m_label, dwave_t_label=dwave_t_label,
        dwave_t_display=dwave_t_display,
        n_label=n_label, m_label=m_label,
        timeout_val=timeout_sec, engine_options=engine_options,
        atlas_checked=atlas_checked, dwave_checked=dwave_checked,
        family_checked=family_checked, edges_checked=edges_checked,
        random_checked=random_checked,
        atlas_disabled=atlas_disabled, dwave_disabled=dwave_disabled,
        family_disabled=family_disabled, edges_disabled=edges_disabled,
        random_disabled=random_disabled,
        family_options=family_options, n_val=n_val, m_val=m_val,
        edges_val=edges_val, debug_checked=debug_checked,
        use_lookup_checked=use_lookup_checked,
        rand_n_val=rand_n_val, rand_m_val=rand_m_val,
        rand_max_hint=rand_max_hint,
    )

    # No graph provided — empty state
    if G_nx is None:
        desc = graph_desc if graph_desc else "none"
        meta = graph_desc if graph_desc else "Select a graph using the controls above."
        no_graph_script = """
          document.getElementById('run-engine-btn').disabled = true;
          document.getElementById('run-engine-btn').style.opacity = '0.4';
        """
        page = HTML.format(
            **ctrl_vars,
            graph_desc=desc, input_meta=meta,
            event_colors_json=json.dumps(EVENT_COLORS),
            input_graph_script=no_graph_script, sse_script="",
        )
        return Response(page, mimetype="text/html")

    # Build input graph metadata
    n = G_nx.number_of_nodes()
    m = G_nx.number_of_edges()
    connected = nx.is_connected(G_nx) if n > 0 else False
    deg_seq = sorted([d for _, d in G_nx.degree()], reverse=True)
    circuit_rank = m - n + (nx.number_connected_components(G_nx) if n > 0 else 0)

    input_meta = (
        f"Nodes: {n} &nbsp; Edges: {m} &nbsp; Connected: {connected}<br>"
        f"Degree seq: {deg_seq}<br>"
        f"Circuit rank: {circuit_rank}"
    )

    # Build input graph vis-network script
    input_graph_script = ""
    if n > 0:
        input_graph_script = small_graph_vis(G_nx, "input-graph")

    # Build SSE script — connects to /stream with the same query params
    sse_script = """
      var qs = window.location.search;
      _es = new EventSource('/stream' + qs);
      var tbody = document.getElementById('timeline-table');
      var evCount = 0;
      var thresholdMs = __THRESHOLD__;
      var MAX_ROWS = 2000;
      var pendingBatch = [];
      var rafScheduled = false;

      // Step-graph viewer state
      var allEvents = [];                  // every event the timeline saw
      var graphEventIdx = [];              // indices (into allEvents) of events with graph_key
      var stepSnapshots = {};              // canonical_key -> {nodes, edges, loops}
      var stepCursor = -1;                 // position within graphEventIdx
      var stepNetwork = null;
      var playTimer = null;

      function renderStep(idx) {
        if (graphEventIdx.length === 0) {
          document.getElementById('step-label').textContent = '(no graph events yet)';
          return;
        }
        if (idx < 0) idx = 0;
        if (idx >= graphEventIdx.length) idx = graphEventIdx.length - 1;
        stepCursor = idx;
        var evIdx = graphEventIdx[idx];
        var ev = allEvents[evIdx];
        var snap = stepSnapshots[ev.graph_key];
        document.getElementById('step-label').textContent =
          'event #' + ev.index + ' · ' + ev.event_type + ' · ' + ev.module +
          ' · ' + (idx + 1) + '/' + graphEventIdx.length;
        if (!snap) {
          document.getElementById('step-graph').innerHTML =
            '<div class="meta" style="padding:12px">snapshot not yet received</div>';
          return;
        }
        var nodes = snap.nodes.map(function(id) {
          return {id: id, label: String(id)};
        });
        var edges = [];
        var eid = 0;
        snap.edges.forEach(function(e) {
          var mult = e[2] || 1;
          for (var k = 0; k < mult; k++) {
            edges.push({id: 'e' + (eid++), from: e[0], to: e[1],
                        smooth: mult > 1 ? {type: 'curvedCW', roundness: 0.2 * k} : false});
          }
        });
        if (snap.loops) {
          snap.loops.forEach(function(l) {
            var mult = l[1] || 1;
            for (var k = 0; k < mult; k++) {
              edges.push({id: 'l' + (eid++), from: l[0], to: l[0],
                          smooth: {type: 'curvedCW', roundness: 0.3 + 0.1 * k}});
            }
          });
        }
        var container = document.getElementById('step-graph');
        container.innerHTML = '';
        stepNetwork = new vis.Network(
          container,
          {nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges)},
          {edges: {smooth: false}, physics: {enabled: true, stabilization: {iterations: 150}}}
        );
        stepNetwork.once('stabilizationIterationsDone', function() {
          stepNetwork.setOptions({physics: {enabled: false}});
        });
      }

      function stepPrev() { renderStep(stepCursor - 1); }
      function stepNext() { renderStep(stepCursor + 1); }
      function stepFirst() { renderStep(0); }
      function stepLast() { renderStep(graphEventIdx.length - 1); }
      function togglePlay() {
        var btn = document.getElementById('step-play');
        if (playTimer) {
          clearInterval(playTimer);
          playTimer = null;
          btn.textContent = 'Play';
          return;
        }
        var intervalMs = parseInt(document.getElementById('step-interval').value) || 500;
        btn.textContent = 'Pause';
        playTimer = setInterval(function() {
          if (stepCursor + 1 >= graphEventIdx.length) {
            clearInterval(playTimer);
            playTimer = null;
            btn.textContent = 'Play';
            return;
          }
          renderStep(stepCursor + 1);
        }, intervalMs);
      }

      function selectTimelineRow(evIndex) {
        var ev = allEvents[evIndex];
        if (!ev) return;
        var targetEvIdx = evIndex;
        if (!ev.graph_key) {
          // Snap to nearest previous event with a graph.
          for (var i = evIndex - 1; i >= 0; i--) {
            if (allEvents[i] && allEvents[i].graph_key) { targetEvIdx = i; break; }
          }
        }
        var pos = graphEventIdx.indexOf(targetEvIdx);
        if (pos >= 0) renderStep(pos);
      }

      document.getElementById('step-first').addEventListener('click', stepFirst);
      document.getElementById('step-prev').addEventListener('click', stepPrev);
      document.getElementById('step-next').addEventListener('click', stepNext);
      document.getElementById('step-last').addEventListener('click', stepLast);
      document.getElementById('step-play').addEventListener('click', togglePlay);

      function flushBatch() {
        rafScheduled = false;
        if (pendingBatch.length === 0) return;

        var frag = document.createDocumentFragment();
        var batch = pendingBatch;
        pendingBatch = [];

        for (var b = 0; b < batch.length; b++) {
          var ev = batch[b];
          allEvents[ev.index] = ev;
          if (ev.graph_key) {
            graphEventIdx.push(ev.index);
          }
          var row = document.createElement('tr');
          row.id = 'ev-' + ev.index;
          row.style.cursor = 'pointer';
          row.setAttribute('data-ev-index', ev.index);

          var durText = '';
          var highlight = false;
          var arrow = '';
          if (ev.gap > 0 && ev.index > 0) {
            durText = ev.gap.toFixed(3) + 's';
            if (ev.gap * 1000 >= thresholdMs) {
              highlight = true;
              arrow = ev.gap >= 1
                ? ' <span style="color:#e65100">&larr; ' + ev.gap.toFixed(1) + 's</span>'
                : ' <span style="color:#e65100">&larr; ' + (ev.gap*1000).toFixed(0) + 'ms</span>';
            }
          }

          if (highlight) row.style.background = '#fff3e0';
          var indent = '';
          for (var d = 0; d < ev.depth; d++) indent += '&nbsp;&nbsp;';
          var graphDot = ev.graph_key
            ? '<span title="' + ev.graph_key.substring(0, 16) + '" style="color:#1565c0">&#x25CF;</span>'
            : '';
          row.innerHTML =
            '<td>' + ev.timestamp + '</td>' +
            '<td>' + durText + '</td>' +
            '<td>' + ev.depth + '</td>' +
            '<td><span class="badge" style="background:' + ev.color + '">' + ev.event_type + '</span></td>' +
            '<td>' + ev.module + '</td>' +
            '<td>' + indent + ev.message + arrow + '</td>' +
            '<td style="text-align:center">' + graphDot + '</td>';
          frag.appendChild(row);
        }

        // Trim old rows if over limit
        while (tbody.rows.length + frag.childNodes.length - 1 > MAX_ROWS && tbody.rows.length > 1) {
          tbody.deleteRow(1);  // keep header
        }

        tbody.appendChild(frag);
        evCount += batch.length;
        document.getElementById('event-count').textContent = '(' + evCount + ' events)';
        document.getElementById('step-graph-count').textContent =
          graphEventIdx.length + ' graph step' + (graphEventIdx.length === 1 ? '' : 's');
        if (stepCursor < 0 && graphEventIdx.length > 0) {
          renderStep(0);
        }

        var scroll = document.getElementById('timeline-scroll');
        scroll.scrollTop = scroll.scrollHeight;
      }

      // Delegate clicks on timeline rows to the step viewer.
      document.getElementById('timeline-table').addEventListener('click', function(e) {
        var row = e.target.closest('tr[data-ev-index]');
        if (!row) return;
        selectTimelineRow(parseInt(row.getAttribute('data-ev-index')));
      });

      function renderMinorsSection(list, containerId, emptyText) {
        var container = document.getElementById(containerId);
        if (!list || list.length === 0) {
          container.innerHTML = '<div class="meta">' + emptyText + '</div>';
          return;
        }
        container.innerHTML = '';
        list.forEach(function(m, i) {
          var card = document.createElement('div');
          card.className = 'minor-card';
          var divId = containerId + '-card-' + i;
          card.innerHTML = '<div class="minor-label">' + m.name + ' (' + m.edges + ' edges)</div>'
            + '<div id="' + divId + '" class="small-graph"></div>';
          container.appendChild(card);
          if (m.nodes) {
            (function(id, nodesJson, edgesJson) {
              setTimeout(function() {
                var net = new vis.Network(
                  document.getElementById(id),
                  {nodes: new vis.DataSet(JSON.parse(nodesJson)),
                   edges: new vis.DataSet(JSON.parse(edgesJson))},
                  opts
                );
                net.fit({padding: 20});
              }, 50);
            })(divId, m.nodes, m.edges_data);
          }
        });
      }

      _es.onmessage = function(msg) {
        var d = JSON.parse(msg.data);

        if (d.type === 'batch') {
          if (d.snapshots) {
            for (var k in d.snapshots) {
              if (d.snapshots.hasOwnProperty(k)) stepSnapshots[k] = d.snapshots[k];
            }
          }
          for (var i = 0; i < d.events.length; i++) {
            pendingBatch.push(d.events[i]);
          }
          if (!rafScheduled) {
            rafScheduled = true;
            requestAnimationFrame(flushBatch);
          }
        }

        else if (d.type === 'done') {
          stopEngine();
          document.getElementById('event-count').textContent = '(' + d.event_count + ' events)';
          document.getElementById('result-container').innerHTML = d.result_html;

          // Summary
          if (d.summary && d.summary.length > 0) {
            var sh = '<table class="summary"><tr><th>EventType</th><th>Count</th><th>Duration</th><th colspan="2">Share</th></tr>';
            d.summary.forEach(function(s) {
              var barW = Math.max(s.pct_num * 1.5, s.pct_num > 0 ? 2 : 0);
              sh += '<tr>'
                + '<td><span class="badge" style="background:' + s.color + '">' + s.event_type + '</span></td>'
                + '<td style="text-align:right">' + s.count.toLocaleString() + '</td>'
                + '<td style="text-align:right">' + s.duration + '</td>'
                + '<td style="text-align:right;width:50px">' + s.pct + '</td>'
                + '<td style="width:160px"><span class="pct-bar" style="width:' + barW + 'px;background:' + s.color + '"></span></td>'
                + '</tr>';
            });
            sh += '</table>';
            document.getElementById('summary-container').innerHTML = sh;
          }

          // Contributing graphs — split by provenance
          var mc = document.getElementById('minors-container');
          var lookupList = d.minors_lookup || d.minors || [];
          var synthList = d.minors_synthesized || [];
          mc.innerHTML =
            '<div class="minor-subsection">' +
              '<h3 style="font-size:12px;margin:0 0 6px 0;color:#2e7d32;">' +
                'From Lookup Table (' + lookupList.length + ')' +
              '</h3>' +
              '<div id="minors-lookup" class="minors-grid"></div>' +
            '</div>' +
            '<div class="minor-subsection" style="margin-top:12px">' +
              '<h3 style="font-size:12px;margin:0 0 6px 0;color:#1565c0;">' +
                'Synthesized During Run (' + synthList.length + ')' +
              '</h3>' +
              '<div id="minors-synthesized" class="minors-grid"></div>' +
            '</div>';
          renderMinorsSection(lookupList, 'minors-lookup',
            'No rainbow table entries used.');
          renderMinorsSection(synthList, 'minors-synthesized',
            'No graphs synthesized from scratch (all hits were in the table).');
        }

        else if (d.type === 'error') {
          stopEngine();
          document.getElementById('result-container').innerHTML =
            '<div class="error-banner">' + d.message + '</div>';
          document.getElementById('minors-container').innerHTML = '';
        }
      };

      _es.onerror = function() {
        stopEngine();
      };
    """.replace("__THRESHOLD__", str(threshold_ms))

    # Build page — use __PLACEHOLDER__ approach to avoid JSON braces in .format()
    page = HTML.format(
        **ctrl_vars,
        graph_desc=graph_desc, input_meta=input_meta,
        event_colors_json=json.dumps(EVENT_COLORS),
        input_graph_script="__INPUT_GRAPH_SCRIPT__",
        sse_script="__SSE_SCRIPT__",
    )
    page = page.replace("__INPUT_GRAPH_SCRIPT__", input_graph_script)
    page = page.replace("__SSE_SCRIPT__", sse_script)
    return Response(page, mimetype="text/html")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5002
    print(f"Tutte Engine Visualizer running at http://localhost:{port}/")
    print(f"  Examples:")
    print(f"    http://localhost:{port}/?atlas=18")
    print(f"    http://localhost:{port}/?atlas=150")
    print(f"    http://localhost:{port}/?edges=0-1,1-2,2-3,3-0,0-2,1-3")
    print(f"    http://localhost:{port}/?source=dwave&dwave_topo=zephyr&dwave_m=1&dwave_t=1")
    app.run(debug=False, port=port, threaded=True)