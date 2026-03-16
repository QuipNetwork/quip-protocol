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

import sys
import os
import json
import time
import re
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from flask import Flask, request, Response

from tutte.graph import Graph
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.synthesis.base import SynthesisResult
from tutte.logs import get_log, reset_log, EventType, LogLevel

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
    from sympy import symbols, factor

    x, y = symbols('x y')
    expr = sum(coeff * x**i * y**j for (i, j), coeff in tutte_poly._coeffs.items())
    factored = factor(expr)

    # If factored form is shorter, use it; otherwise show expanded
    fact_str = str(factored)
    exp_str = str(expr)
    display = fact_str if len(fact_str) < len(exp_str) else exp_str

    return poly_to_html(display)


def parse_graph(args) -> tuple:
    """Parse URL parameters into a (nx.Graph, description) tuple."""
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
                t = dwave_t if dwave_t is not None else dwave_m
                G = dnx.chimera_graph(dwave_m, t)
                return G, f"Chimera C({dwave_m},{t})"
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
            <span id="dwave-t-wrap"><span id="dwave-t-label">{dwave_t_label}</span>=<input type="number" name="dwave_t" value="{dwave_t_val}" placeholder="1" min="1" style="width:50px" {dwave_disabled}></span>
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
            <input type="number" name="timeout" value="{timeout_val}" min="1" max="600" style="width:60px">s
          </div>
          <div class="ctrl-row">
            <label><input type="checkbox" name="debug" value="1" {debug_checked}> Debug logging</label>
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
        'chimera': ['rows of tiles', 'shore size'],
      }};
      function updateDwave() {{
        var topo = dsel.value;
        // Zephyr: m, t; Pegasus: m only (min 2); Chimera: m, t
        var showT = (topo === 'zephyr' || topo === 'chimera');
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
    <h2>Timeline <span id="event-count"></span></h2>
    <div class="timeline-scroll" id="timeline-scroll">
      <table class="timeline" id="timeline-table">
        <tr><th>Time</th><th>Duration</th><th>D</th><th>Type</th><th>Module</th><th>Message</th></tr>
      </table>
    </div>
  </div>

  <script>
    var EVENT_COLORS = {event_colors_json};
    var opts = {{ edges: {{ smooth: false }}, physics: {{ enabled: false }} }};
    var _es = null;  // current EventSource

    window.addEventListener('DOMContentLoaded', function() {{
      // Render input graph
      {input_graph_script}
    }});

    function stopEngine() {{
      if (_es) {{ _es.close(); _es = null; }}
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
      document.getElementById('result-container').innerHTML = '<div class="meta"><span class="spinner"></span>Running engine...</div>';
      document.getElementById('minors-container').innerHTML = '<div class="meta"><span class="spinner"></span>Waiting for engine...</div>';
      document.getElementById('summary-container').innerHTML = '';

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


def _run_engine_thread(graph, table, timeout_sec, engine_type):
    """Run the engine in a thread, storing result in _engine_state."""
    global _engine_state
    result_holder = [None]
    error_holder = [None]

    def target():
        try:
            if engine_type == "hybrid":
                from tutte.synthesis.hybrid import HybridSynthesisEngine
                engine = HybridSynthesisEngine(table=table)
                hybrid_result = engine.synthesize(graph)
                result_holder[0] = SynthesisResult(
                    polynomial=hybrid_result.polynomial,
                    recipe=hybrid_result.recipe,
                    verified=hybrid_result.verified,
                    method=hybrid_result.method,
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

    G_nx, graph_desc = parse_graph(request.args)
    if G_nx is None:
        def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': graph_desc or 'No graph'})}\n\n"
        return Response(error_stream(), mimetype="text/event-stream")

    graph = Graph.from_networkx(G_nx)
    table = load_default_table()

    # Reset log and set min_level based on debug toggle
    reset_log()
    get_log().min_level = LogLevel.DEBUG if debug else LogLevel.INFO
    global _engine_state
    with _engine_lock:
        _engine_state = {
            "running": True, "result": None, "error": None,
            "timed_out": False, "elapsed": 0.0, "done": False,
        }

    # Start engine in background thread
    engine_thread = threading.Thread(
        target=_run_engine_thread,
        args=(graph, table, timeout_sec, engine_type),
        daemon=True,
    )
    engine_thread.start()

    def _serialize_batch(new_events, start_idx, prev_timestamp):
        """Serialize a list of new events into a single SSE batch message.

        Returns (json_str, last_timestamp).
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
            })
            prev_timestamp = ev.timestamp
        return batch, prev_timestamp

    def event_stream():
        log = get_log()
        sent_count = 0
        prev_timestamp = None
        poll_interval = 0.05  # 50ms

        while True:
            new_events = log.events_since(sent_count)

            if new_events:
                batch, prev_timestamp = _serialize_batch(
                    new_events, sent_count, prev_timestamp
                )
                sent_count += len(new_events)
                yield f"data: {json.dumps({'type': 'batch', 'events': batch})}\n\n"

            # Check if engine is done
            with _engine_lock:
                done = _engine_state["done"]

            if done:
                # Send any remaining events
                remaining = log.events_since(sent_count)
                if remaining:
                    batch, prev_timestamp = _serialize_batch(
                        remaining, sent_count, prev_timestamp
                    )
                    sent_count += len(remaining)
                    yield f"data: {json.dumps({'type': 'batch', 'events': batch})}\n\n"

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
                elif error:
                    final["result_html"] = (
                        f'<div class="error-banner">Engine error: {error}</div>'
                    )
                    final["minors"] = []
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
                    # Build contributing graphs (minors)
                    minors_list = []
                    if result.minors_used:
                        for key in sorted(result.minors_used):
                            entry = table.get_entry_by_key(key)
                            if entry is None:
                                continue
                            minor_info = {"name": entry.name, "edges": entry.edge_count}
                            minor_nx = graph_from_entry(entry)
                            if minor_nx is not None:
                                nodes_json, edges_json = vis_data_json(minor_nx)
                                minor_info["nodes"] = nodes_json
                                minor_info["edges_data"] = edges_json
                            minors_list.append(minor_info)
                            if len(minors_list) >= 6:
                                break
                    final["minors"] = minors_list
                else:
                    final["result_html"] = (
                        '<div class="error-banner">Engine returned no result.</div>'
                    )
                    final["minors"] = []

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
    for topo, label in [("zephyr", "Zephyr Z(m, t)"), ("pegasus", "Pegasus P(m)"), ("chimera", "Chimera C(m, t)")]:
        sel = " selected" if topo == dwave_topo_val else ""
        dwave_topo_options += f'<option value="{topo}"{sel}>{label}</option>'

    G_nx, graph_desc = parse_graph(request.args)

    debug_checked = "checked" if debug else ""

    # Compute server-side labels for D-Wave params
    _dwave_labels = {
        "zephyr": ("grid parameter", "tile parameter"),
        "pegasus": ("size parameter", ""),
        "chimera": ("rows of tiles", "shore size"),
    }
    dwave_m_label, dwave_t_label = _dwave_labels.get(dwave_topo_val, ("m", "t"))
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

      function flushBatch() {
        rafScheduled = false;
        if (pendingBatch.length === 0) return;

        var frag = document.createDocumentFragment();
        var batch = pendingBatch;
        pendingBatch = [];

        for (var b = 0; b < batch.length; b++) {
          var ev = batch[b];
          var row = document.createElement('tr');
          row.id = 'ev-' + ev.index;

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
          row.innerHTML =
            '<td>' + ev.timestamp + '</td>' +
            '<td>' + durText + '</td>' +
            '<td>' + ev.depth + '</td>' +
            '<td><span class="badge" style="background:' + ev.color + '">' + ev.event_type + '</span></td>' +
            '<td>' + ev.module + '</td>' +
            '<td>' + indent + ev.message + arrow + '</td>';
          frag.appendChild(row);
        }

        // Trim old rows if over limit
        while (tbody.rows.length + frag.childNodes.length - 1 > MAX_ROWS && tbody.rows.length > 1) {
          tbody.deleteRow(1);  // keep header
        }

        tbody.appendChild(frag);
        evCount += batch.length;
        document.getElementById('event-count').textContent = '(' + evCount + ' events)';

        var scroll = document.getElementById('timeline-scroll');
        scroll.scrollTop = scroll.scrollHeight;
      }

      _es.onmessage = function(msg) {
        var d = JSON.parse(msg.data);

        if (d.type === 'batch') {
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

          // Contributing graphs
          var mc = document.getElementById('minors-container');
          if (d.minors && d.minors.length > 0) {
            mc.innerHTML = '';
            d.minors.forEach(function(m, i) {
              var card = document.createElement('div');
              card.className = 'minor-card';
              card.innerHTML = '<div class="minor-label">' + m.name + ' (' + m.edges + ' edges)</div>'
                + '<div id="minor-' + i + '" class="small-graph"></div>';
              mc.appendChild(card);
              // Render graph if data available
              if (m.nodes) {
                setTimeout(function() {
                  var net = new vis.Network(
                    document.getElementById('minor-' + i),
                    {nodes: new vis.DataSet(JSON.parse(m.nodes)), edges: new vis.DataSet(JSON.parse(m.edges_data))},
                    opts
                  );
                  net.fit({padding: 20});
                }, 50);
              }
            });
          } else {
            mc.innerHTML = '<div class="meta">No rainbow table entries used.</div>';
          }
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