# Engine Workflow Primer

A self-contained walkthrough of the Tutte synthesis engine for readers who
have studied high school algebra but have not necessarily studied graph
theory. We start from the definition of a graph, build up to the Tutte
polynomial, then walk every stage of the engine pipeline (`tutte/docs/README.md`)
in order — defining each piece of vocabulary as it appears.

The goal is to give a reader enough vocabulary and enough intuition to read
the per-technique docs in `tutte/docs/` and to follow the source code in
`tutte/`. Each section ends with pointers to the relevant in-tree docs and
to the published papers that the technique builds on.

---

## Part I — Vocabulary

### 1. What is a graph?

A **graph** is a pair `G = (V, E)`, where:

- **`V`** is a finite set of _vertices_ (also called _nodes_). For us, vertices
  are usually integers `0, 1, 2, …`.
- **`E`** is a set of _edges_. An edge is an unordered pair `{u, v}` of two
  distinct vertices. We write it `(u, v)` with the convention that `u < v`.

Pictorially, a graph is dots (vertices) connected by lines (edges). The graph
`G = ({0, 1, 2}, {(0, 1), (1, 2), (0, 2)})` is a triangle.

In our codebase the type is `Graph(nodes, edges)` in `tutte/graph.py`.

A **MultiGraph** allows the same edge to appear more than once (a _parallel
edge_) and allows _loops_ — edges from a vertex to itself. The class
`MultiGraph` in `tutte/graph.py` stores edges as a multiset
`edge_counts: Dict[(u, v), int]`.

### 2. Adjacency, degree, neighbors

Two vertices are **adjacent** if they share an edge. The **degree** of a
vertex `v`, written `deg(v)`, is the number of edges incident to `v`. The
**neighbors** of `v` is the set `N(v) = { u : (u, v) ∈ E }`.

### 3. Walks, paths, cycles

A **walk** is a sequence of vertices `v_0, v_1, …, v_k` where each consecutive
pair `(v_i, v_{i+1})` is an edge.

- A walk with no repeated vertices is a **path**.
- A walk where `v_0 = v_k` (start = end) and no other vertex repeats is a
  **cycle**.

The **path graph on n vertices**, `P_n`, is just a single path through `n`
vertices. The **cycle graph on n vertices**, `C_n`, is a single cycle.

### 4. Connectedness, components, disconnected graphs

A graph is **connected** if you can reach any vertex from any other vertex by
walking along edges.

If a graph is not connected, it splits into several maximal connected pieces
called **connected components**. We sometimes call a graph **disconnected**
to emphasize that it has more than one component.

When the engine sees a disconnected graph it factors the polynomial into a
product (one factor per component) — see [Stage 4](#stage-4--disconnected-factorization).

### 5. Subgraphs, spanning subgraphs, induced subgraphs

A **subgraph** of `G = (V, E)` is a graph `G' = (V', E')` with `V' ⊆ V` and
`E' ⊆ E` (and every edge of `G'` only uses vertices in `V'`).

- **Spanning subgraph**: `V' = V`. Same vertices, possibly fewer edges.
- **Induced subgraph on `V'`**: take exactly the edges of `G` whose
  endpoints both lie in `V'`. Notation: `G[V']`.

### 6. Trees, spanning trees, forests

A **tree** is a connected graph with no cycles. A graph with no cycles (but
possibly disconnected) is a **forest** — every connected component is a tree.

A **spanning tree** of `G` is a spanning subgraph that is also a tree. Every
connected graph has at least one spanning tree.

A celebrated result, **Kirchhoff's matrix-tree theorem** (Kirchhoff, 1847),
counts spanning trees as a determinant of the graph Laplacian. The engine
uses this as a sanity check: `T(1, 1) =` the spanning tree count, computable
in `O(n³)` via determinant, even for graphs whose Tutte polynomial we cannot
afford to compute. See `tutte/validation.py::count_spanning_trees_kirchhoff`.

### 7. Cut vertices and bridges

- A **cut vertex** (or _articulation point_) is a vertex whose removal
  disconnects the graph. Equivalently: removing it increases the number of
  connected components.
- A **bridge** is an edge whose removal disconnects the graph.

In a tree, every edge is a bridge and every non-leaf vertex is a cut vertex.
In a cycle, none are. Cut vertices and bridges are _the_ canonical structural
"weak points" of a graph; the engine has a fast path that splits at every
cut vertex (Stage 5).

### 8. Series, parallel, and "operations on graphs"

Two edges are in **series** if they share a vertex of degree 2 with no
other neighbors — like two resistors in series in a circuit. Replacing a
degree-2 vertex `v` and its two edges `(u, v), (v, w)` by a single edge
`(u, w)` is a **series reduction**.

Two edges between the same pair of vertices `u` and `v` are **parallel**.
Replacing them by a single edge is a **parallel reduction**.

A graph that can be reduced to a single edge by repeated series and parallel
reductions is called **series-parallel** (often abbreviated **SP**).

Other operations:

- **Edge deletion**, written `G − e`: remove edge `e`, keep its endpoints.
- **Edge contraction**, written `G / e`: shrink edge `e` to a point — merge
  its two endpoints into one (re-routing all their other edges to the merged
  vertex), and delete `e` itself. If the contraction creates parallel edges
  or loops, those are kept (they show up in the multigraph view).
- **Disjoint union** of `G_1` and `G_2`: take both graphs side by side, with
  no edges between them. Notation: `G_1 ∪ G_2`.

### 9. Cliques and complete graphs

A **clique** is a set of vertices that are _pairwise adjacent_ — every two
of them share an edge. The **complete graph on `n` vertices**, `K_n`, has all
possible edges: there is an edge between every pair of distinct vertices, so
`|E(K_n)| = n(n−1)/2`.

A **k-clique** is a clique of size `k`.

The **complete bipartite graph** `K_{a,b}` has `a + b` vertices split into
two sides (size `a` and size `b`) with every left vertex adjacent to every
right vertex (and no edges within a side).

### 10. Isomorphism

Two graphs `G_1 = (V_1, E_1)` and `G_2 = (V_2, E_2)` are **isomorphic** if
there is a bijection `σ: V_1 → V_2` that maps edges to edges:
`(u, v) ∈ E_1 ⇔ (σ(u), σ(v)) ∈ E_2`. Informally: same shape, just relabeled
vertices.

A bijection `σ: V → V` from a graph to itself that preserves edges is an
**automorphism**. The set of all automorphisms forms the **automorphism
group** `Aut(G)`. For `K_n`, `Aut(K_n)` is the full symmetric group `S_n`
(any permutation of the vertices is an automorphism). Automorphism groups
matter to us because they let us identify _equivalent_ sub-problems and
cache them once — see Stage 6.8 (cell-quotient DPs).

---

## Part II — The Tutte Polynomial

### 11. Definition by deletion-contraction

The **Tutte polynomial** `T(G; x, y)` is a two-variable polynomial defined
recursively. It was introduced by W. T. Tutte in 1947 ("A ring in graph
theory", _Math. Proc. Cambridge Philos. Soc._) building on earlier work by
Hassler Whitney (1932).

The recursive definition has three base cases plus one recursive rule:

| Case                                                    | Rule                         |
| ------------------------------------------------------- | ---------------------------- |
| **Empty edges** (the graph has only vertices, no edges) | `T(G) = 1`                   |
| **`e` is a bridge**                                     | `T(G) = x · T(G − e)`        |
| **`e` is a loop**                                       | `T(G) = y · T(G − e)`        |
| **`e` is neither a bridge nor a loop**                  | `T(G) = T(G − e) + T(G / e)` |

That's it. Every Tutte polynomial can be computed from the empty graph by
applying these four rules to one edge at a time.

A nontrivial theorem says: this definition does not depend on the order in
which you process the edges. The same polynomial comes out either way.

### 12. Evaluations and what they mean

The Tutte polynomial encodes a remarkable amount of structural information:

| Evaluation | Counts                          |
| ---------- | ------------------------------- |
| `T(1, 1)`  | spanning trees                  |
| `T(2, 1)`  | spanning forests                |
| `T(1, 2)`  | connected spanning subgraphs    |
| `T(2, 0)`  | acyclic orientations            |
| `T(0, 2)`  | strongly connected orientations |

For example, `T(C_3; x, y) = x² + x + y`, so `T(C_3; 1, 1) = 1 + 1 + 1 = 3`,
which is the number of spanning trees of a triangle (delete any one edge,
get a path on 3 vertices).

The chromatic polynomial and reliability polynomial are also evaluations of
the Tutte polynomial, after a change of variable.

### 13. Why it matters here

The Quip Protocol uses the Tutte polynomial as a **structural difficulty
measure** for Ising-model proof-of-work — see `tutte/docs/00_motivation.md`
for the full picture. The short story: computing `T(G)` is `#P`-hard in
general (Jaeger, Vertigan & Welsh, 1990), but verifying many of its
properties is cheap, which is what we want from a proof-of-work mechanism.

The engine's job is to compute `T(G)` as fast as possible for the kinds of
graphs that come out of quantum hardware topologies (D-Wave Chimera, Pegasus,
Zephyr; IBM heavy-hex; etc.). The pipeline is a sequence of fast paths,
each cheaper than the next, each catching a structural shape that the engine
knows how to handle without falling all the way to brute-force
deletion-contraction.

---

## Part III — The Pipeline

The engine in `tutte/synthesis/engine.py` tries a sequence of techniques, in
roughly increasing order of cost. Each technique either returns a
polynomial and stops, or returns `None` (a "fall through") and the next
technique gets a turn.

The pipeline diagram lives in `tutte/docs/README.md`. Each stage below has
a one-paragraph definition + a pointer to the matching `tutte/docs/` file.

### Stage 1 — Family Recognition

**Doc**: `tutte/docs/01_family_recognition.md`. **Cost**: `O(n + m)`.

Some graph families have a **closed-form** Tutte polynomial — a single
formula in the parameter `n` (number of vertices) or `(m, n)` (rows, columns).
Examples:

- A tree on `n` vertices: `T = x^{n−1}`.
- A cycle on `n` vertices: `T = x^{n−1} + x^{n−2} + … + x + y`.
- Wheels, ladders, fans, prisms, books, gears, Möbius ladders, grids: all
  have known closed forms or simple constant-coefficient recurrences.

The recognition module checks "is this graph one of these named shapes?" by
looking at degree sequences and small structural fingerprints. If yes, we
return the polynomial without ever doing deletion-contraction. This is the
fastest path in the engine.

### Stage 2 — Rainbow Table Lookup

**Doc**: `tutte/docs/02_rainbow_table_lookup.md`. **Cost**: `O(n² × d)`
dominated by a _canonical key_ computation.

Given two isomorphic graphs `G_1` and `G_2`, they have the same Tutte
polynomial. So if we have already computed `T(G_1)` and stored it, we can
look up `T(G_2)` instantly — _if_ we have a way to recognize that two graphs
are isomorphic.

The classical isomorphism test is exponential in the worst case, so the
engine uses a **canonical key**: a function that maps any graph to a string
of bytes such that isomorphic graphs always get the same string. The key
is built from the **Weisfeiler-Leman algorithm** (Weisfeiler & Leman, 1968),
which iteratively refines vertex colors based on neighbor colors. The
result is unique up to isomorphism for almost every graph. (There are
pathological exceptions; we're fine in practice.)

The "rainbow table" itself (`tutte/data/lookup_table.bin`) is a precomputed
dictionary from canonical keys to Tutte polynomials. It includes the small
named graphs (K*n for small n, Petersen, K*{a,b}, etc.) and is grown over
time by saving every successfully synthesized polynomial.

### Stage 3 — Base Cases

**Doc**: `tutte/docs/03_base_cases.md`. **Cost**: `O(1)`.

Two trivial cases:

- Empty graph (no edges): `T = 1`.
- Single edge graph: `T = x`.

Self-loops contribute a factor of `y` per loop; parallel edges fold into
the parallel-edge formula at the multigraph level.

### Stage 4 — Disconnected Factorization

**Doc**: `tutte/docs/04_disconnected_factorization.md`. **Cost**: `O(n + m)`

- recursive synthesis on each component.

A disjoint union factors:

```
T(G_1 ∪ G_2) = T(G_1) · T(G_2)
```

So the engine finds the connected components, recursively synthesizes each,
and multiplies. This reduces a possibly large disconnected graph to several
smaller connected sub-problems.

### Stage 5 — Cut Vertex Factorization

**Doc**: `tutte/docs/05_cut_vertex_factorization.md`. **Cost**: `O(n + m)`

- recursive synthesis per **block**.

If `v` is a _cut vertex_ (Stage I.7), then removing `v` disconnects `G` into
at least two components. The graph splits into "blocks" — maximal subgraphs
with no cut vertex of their own — that share `v` at the seams. The Tutte
polynomial multiplies across blocks:

```
T(G_1 · G_2) = T(G_1) · T(G_2)        (where G_1, G_2 share exactly one vertex)
```

This is a special case of the general 1-sum identity. The engine finds all
cut vertices in `O(n + m)` (Tarjan, 1972) and recursively synthesizes each
block.

### Stage 6 — Treewidth Dynamic Programming

**Docs**: `tutte/docs/06_treewidth_dp.md`. **Cost**: `O(2^tw × n)` via a
C extension.

This is where the first heavy machinery shows up. We need three new pieces
of vocabulary.

#### 6a. Tree decomposition

A **tree decomposition** of `G` is a tree `T` plus a _bag_ `B_t ⊆ V(G)` for
each node `t` of `T`, such that:

1. Every vertex of `G` appears in at least one bag.
2. Every edge of `G` has both endpoints in at least one common bag.
3. For any vertex `v ∈ V(G)`, the bags containing `v` form a connected
   subtree of `T`.

You can think of the tree decomposition as "cutting `G` into overlapping
pieces (bags) and arranging the pieces in a tree".

#### 6b. Treewidth

The **width** of a tree decomposition is `max_t |B_t| − 1` — one less than
the size of the biggest bag. The **treewidth** of `G`, written `tw(G)`, is
the minimum width over all tree decompositions of `G`. Trees themselves
have treewidth 1; cycles have treewidth 2; `K_n` has treewidth `n − 1`.

Treewidth is a profound structural parameter introduced by Robertson &
Seymour (1984) in their Graph Minors series. Many `NP`-hard problems become
tractable on graphs of bounded treewidth, including computing the Tutte
polynomial (Andrzejak, 1998; Noble, 1998).

#### 6c. The DP itself

Given a tree decomposition of width `w`, the algorithm walks the tree from
leaves to root. For each bag `B_t`, it maintains a table indexed by _partial
states_ — partitions of the bag's vertices encoding which subgraph
components are connected — and contributing coefficients to the running
polynomial. The table size is bounded by `Bell(w + 1)`, where `Bell(n)` is
the _Bell number_ (the number of set partitions of an `n`-element set).

The engine uses a C extension (`tutte/graphs/_treewidth_c.py`) for speed,
gated to `treewidth ≤ 11` (Bell(12) ≈ 4 million states is the practical
ceiling). For graphs with too many edges to fit in `int64` arithmetic, the
DP runs with **Chinese Remainder Theorem (CRT)** encoding — see Stage 6e.

#### 6d. Why it doesn't always work

For graphs with `tw > 11`, the bag-DP table grows beyond practical memory.

#### 6e. Chinese Remainder Theorem (CRT)

The classical **Chinese Remainder Theorem** (Sun Tzu, ~3rd century CE; modern
formulation dates to Gauss, 1801) says: if `m_1, m_2, …, m_k` are pairwise
coprime integers, then for any `(a_1, …, a_k)` there is a unique `x` modulo
`M = m_1 · m_2 · … · m_k` with `x ≡ a_i (mod m_i)` for each `i`.

We use this for _modular polynomial arithmetic_. Tutte polynomial coefficients
can be enormous — for a graph with `m` edges, individual coefficients can
exceed `2^m / sqrt(m)`, which overflows `int64` past about 63 edges. Instead
of using arbitrary-precision integers (slow), the engine computes the
polynomial modulo several small primes `p_1, p_2, …` simultaneously, then
reconstructs the true coefficients via CRT at the end. See
`tutte/graphs/_treewidth_c.py`'s `treewidth_tutte_dp_modular` path.

### Stage 6.5 — Cotree DP

**Doc**: `tutte/docs/06_1_cotree_dp.md`. **Cost**: `exp(O(n^{2/3}))` —
sub-exponential.

A few new pieces of vocabulary first.

#### Cograph

A **cograph** (short for "complement-reducible graph") is built from single
vertices using only two operations:

- **Disjoint union** (`∪`): take `G_1 ∪ G_2` side by side.
- **Complete join** (`⊗`): take `G_1` and `G_2` and add every possible edge
  between a vertex of `G_1` and a vertex of `G_2`.

`K_n`, `K_{a,b}`, threshold graphs, and many "very symmetric" graphs are
cographs. The structural recursion of a cograph is captured by its **cotree**:
a binary tree whose leaves are vertices and whose internal nodes are
labelled `∪` or `⊗`.

#### `P_4`-free characterization

A theorem of Corneil, Lerchs & Stewart Burlingham (1981) says: a graph is a
cograph if and only if it contains no induced **`P_4`** — no induced subgraph
isomorphic to a path on 4 vertices. So checking "is it a cograph?" is a
search for induced `P_4`s.

#### Why it helps for the Tutte polynomial

Giménez, Hliněný & Noy (2006) showed that the Tutte polynomial of a cograph
on `n` vertices can be computed in time `exp(O(n^{2/3}))` — sub-exponential.
This beats the brute-force `O(2^m)` deletion-contraction by a wide margin
for graphs like `K_15`, `K_{8,8}`, and large threshold graphs that are
out of treewidth_dp's range.

The engine's cotree DP module lives in `tutte/cotree_dp/`. The two-step
recurrence (`disjoint_union_subgraph_combine` and `complete_union_subgraph_combine`)
walks the cotree bottom-up.

### Stage 6.6 — Almost-Cograph DP

**Doc**: `tutte/docs/06_2_almost_cograph_dp.md`. **Cost**: 1 cotree DP +
`O(|A|)` recursive syntheses, where `|A|` is the number of _anomaly edges_.

A natural generalization of "is this graph a cograph?" is "could it be a
cograph if we removed a few edges?". Given a graph `G`, an **anomaly edge
set** `A ⊆ E(G)` is a set of edges such that `G − A` is a cograph. The
engine uses a greedy `P_4`-elimination procedure to find a small `A` (capped
at `|A| ≤ 16` for tractability).

If `|A|` is small enough, we apply the **chord rule** (defined in Stage 8
below) to peel off one anomaly edge at a time, each step calling cotree DP
on the now-cograph residual. This trades a small exponential `2^|A|` for the
sub-exponential cotree DP, which is favorable when `|A|` ≤ 16.

### Stage 6.7 — Rooted Tutte Framework (theory reference)

**Doc**: `tutte/docs/06_3_rooted_tutte_framework.md`.

Not a pipeline step. A theory reference for the cell-quotient DPs that
follow. The **rooted Tutte polynomial** generalizes `T(G)` by tracking
which subsets of a marked **boundary set** `S ⊆ V(G)` end up in the same
component of each spanning subgraph. Formally:

```
T_rooted(G, S)[P] = Σ_{A ⊆ E} (x − 1)^{r(E) − r(A)} (y − 1)^{|A| − r(A)}
                   restricted to spanning subgraphs A whose boundary
                   component-partition equals P
```

where `P` is a set partition of `S`. The standard `T(G)` is the sum over
all `P`. Rooted Tutte polynomials carry strictly more information than
standard ones — exactly enough to compose two graphs at a shared vertex
boundary via vertex-sum convolution, which is what cycle/grid DPs exploit.

The classical 2-vertex-sum identity (Brylawski, 1971; Oxley, 2011 §11.4)
is a special case: for graphs sharing exactly two vertices,
`T(G_1 ⊕_2 G_2) = T(G_1) · T(G_2) / T(K_2)`. Rooted Tutte machinery
generalizes this to k-vertex boundaries with arbitrary partition states.

### Stage 6.8 — Cell-Quotient Cycle DP

**Doc**: `tutte/docs/06_4_cell_quotient_cycle_dp.md`. **Cost**: `O(n_cells × Bell(W)² × poly²)`.

Some new vocabulary.

#### Cells, anchors, junctions

A **cell** is a small subgraph that appears (often repeatedly) as a building
block of `G`. Concretely: if we partition `V(G)` into pieces `C_0, C_1, …,
C_{k−1}` such that each `G[C_i]` is isomorphic to a known small graph (a
"cell template"), and the only edges between cells come through specific
**anchor** vertices in each cell, then `G` is **cell-decomposable**.

A **junction** between cells `C_i` and `C_j` is the set of edges between
them. The pattern of which-anchor-to-which-anchor is the **junction template**.
For D-Wave Chimera, each cell is `K*{4,4}`and each junction is a
4-edge perfect matching`M_4`.

#### Cell-quotient graph

Collapse each cell to a single node, keep one edge for each junction. The
resulting graph is the **cell-quotient graph** (also called the _quotient
graph_). For Cm_m it is an `m × m` grid; for `Z(1,1)` it is a 4-cycle.

When the cell-quotient is a _simple cycle_, the engine has a specialized DP
that walks around the cycle, composing rooted-Tutte tables at each junction
via vertex-sum convolution, then closing the cycle with an identification
step. This is implemented in `tutte/roots/cell_quotient_cycle.py` and
documented in detail in `tutte/docs/06_4_cell_quotient_cycle_dp.md`.

### Stage 6.9 — Cell-Quotient Grid DP

**Doc**: `tutte/docs/06_5_cell_quotient_grid_dp.md`.

Same idea as 6.8 but for grid topologies (`rows × cols`). Compose row by
row, then close column-wise junctions. Implemented in
`tutte/roots/cell_quotient_grid.py`. The interleaved Hamiltonian-path
variant lives in `tutte/roots/cell_quotient_interleaved.py`.

### Stage 7 — k-Sum Decomposition (chord rule)

**Doc**: `tutte/docs/07_k_sum_decomposition.md`. **Cost**: `1 + C(k, 2)`
full syntheses.

#### Vertex separator

A **k-vertex separator** is a set of `k` vertices `S ⊆ V` whose removal
disconnects the graph: `G − S` has more than one connected component.
Treewidth, defined above, is essentially the minimum size of the worst
separator over a recursive decomposition.

#### Parallel connection

If `G` has a `k`-vertex separator `S` such that the two sides `A` and `B`
together with `S` cover the graph, then we can think of `G` as the **k-sum**
or **parallel connection** `G_1 ⊕_k G_2`, where:

- `G_1 = G[A ∪ S]` (the induced subgraph on side A plus separator).
- `G_2 = G[B ∪ S]` (side B plus separator).
- The two pieces share exactly the `k` vertices of `S`.

Brylawski's 2-sum identity (Stage 6.7) handles `k = 2` with a clean
algebraic factorization. For `k ≥ 3`, no such polynomial-divisor closed form
exists in general.

The engine instead uses the **chord rule**: add the missing `K_k` clique
edges to the separator (call the resulting graph `PC` for "parallel
completion"), then peel them off one at a time using deletion-contraction:

```
T(G) = T(PC) − Σ_i T((PC − e_1 − … − e_{i−1}) / e_i)
```

For `k = 7`, that's `1 + C(7, 2) = 22` syntheses on smaller graphs. Each
synthesis is recursive, so this is mostly worthwhile when those smaller
recursive calls hit the rainbow table or treewidth_dp.

The full mathematical justification appears in
`tutte/docs/08_2_chord_rule_formalization.md`. The chord rule replaced an
older matroid-theoretic implementation (Bonin & de Mier, 2004) in April 2026
because it is mathematically simpler and computationally competitive.

### Stage 8 — Hierarchical Tiling (chord rule)

**Doc**: `tutte/docs/08_hierarchical_tiling.md`. **Cost**: `O(chord_count)`
full syntheses.

When a graph is cell-decomposable (Stage 6.8) but the cell-quotient is _not_
a simple cycle (or grid), the engine uses a more general approach: the
**boundary quotient** formula plus chord recursion.

#### Boundary

The **boundary** of a cell `C_i` is the set of vertices in `C_i` that
participate in at least one inter-cell edge. The **boundary subgraph** `B`
is the induced subgraph on the union of all boundaries plus all inter-cell
edges plus intra-cell edges between boundary vertices.

#### Chord vs bridge in the inter-cell graph

When the inter-cell graph has cyclic structure, some edges are **chords**
(closing cycles) and some are **bridges** (tree edges). The chord rule peels
off the chords one at a time:

```
T(G) = T(G − all chords) + Σ_i T((G − chord_1 − … − chord_{i−1}) / chord_i)
```

Each chord costs one full synthesis on a smaller graph. When the smaller
graphs hit the rainbow table or treewidth_dp, this works well.

If there are no chords (the inter-cell graph is a tree), there's a clean
algebraic shortcut — the **boundary quotient formula**:

```
T(G) = (∏_i T(C_i)) · T(B) / (∏_i T(B_i))
```

Implemented in `tutte/graphs/k_sum.py:boundary_quotient_tutte`.

### Stage 9 — Creation-Expansion-Join (CEJ)

**Doc**: `tutte/docs/09_creation_expansion_join.md`. **Cost**:
`O(chords × synthesis_cost)`.

The final fallback. Pick a spanning tree of `G` (always exists for
connected graphs), compute its Tutte polynomial trivially (`x^{n − 1}`),
then add the non-tree edges (called **chords** here, in the spanning-tree
sense) one at a time using the chord rule. Each chord-addition step is a
deletion-contraction:

```
T(G + e) = T(G) + T(G / e_endpoints_merged)
```

This is essentially brute-force deletion-contraction with the optimization
that the spanning-tree skeleton has a closed-form polynomial. It always
terminates and always gives the right answer; it's slow only because it
makes one recursive call per chord without any structural shortcuts.

---

## Part IV — Closed-form Algebraic Shortcuts

Above the pipeline, the engine has a handful of **closed-form algebraic
formulas** that fire when applicable, replacing entire DP recursions with a
single polynomial multiplication. These are the "wins" that let the engine
beat raw deletion-contraction by orders of magnitude on structured inputs.

### Unified formula

```
T(G) = (∏_i T(cell_i)) · T(H)
```

holds whenever `G` decomposes into cells joined by single-vertex-pair
connections (each cell-pair has at most one unique inter-cell edge endpoint
pair). `H` is the _cell-topology graph_ — one node per cell, one edge per
inter-cell edge, with parallel edges in `H` mirroring parallel inter-cell
edges in `G`.

This formula generalizes the cut-vertex (Stage 5), bridge, and parallel-edge
factorizations into one statement.

### k-Matching formula

```
T(G_1 + M_k + G_2) = (x + k − 1) · T(G_1) · T(G_2)
                     + Σ_{j=2}^{k} C(k, j) · T(G_1 ⊕_j G_2)
```

holds for vertex-transitive cells joined by a `k`-edge perfect matching `M_k`.
Validated on K*3, K_4, K_5, \*\*and K*{4,4} = D-Wave Chimera cell\*\*, for
`k = 1..4`. This is the formula that lets the engine handle Cm*2 in ≈50s
instead of hours: each Cm_2 cell-pair junction is exactly an `M_4` matching
on `K*{4,4}` cells.

Engine wiring: `tutte/synthesis/engine.py:_try_formula_shortcircuit` (step 7
in the pipeline diagram).

---

## Part V — Things to read next

In approximate order from least to most technical:

1. `tutte/docs/00_motivation.md` — why we care about all this.
2. `tutte/docs/README.md` — pipeline overview with the flowchart.
3. `tutte/docs/01_family_recognition.md` through `09_creation_expansion_join.md`
   — one per pipeline stage, in order.
4. `tutte/roots/README.md` — module overview for the cell-quotient DPs.
5. `tutte/research/multivariate_tutte_results.md` — Phase 18.E.1 NEGATIVE
   result on Sokal multivariate `Z`. Shows what it looks like when a
   research direction _doesn't_ pan out.
6. `tutte/research/literature_search_2026.md` — catalog of recent
   (2024-2025) Tutte polynomial papers organized by direction.
7. `tutte/research/cm3_interleaved_attempt.md` — case study of Cm₃ being
   structurally walled by current architecture.

---

## Bibliography

References are listed in roughly the order they appear in the document
above. Where a result is folklore or has no canonical citation, the
attribution gives the technique's typical historical origin.

### Foundational

- Whitney, H. (1932). "The coloring of graphs." _Annals of Mathematics_.
  _(rank polynomial; precursor to the Tutte polynomial)_
- Tutte, W. T. (1947). "A ring in graph theory." _Math. Proc. Cambridge
  Philos. Soc._ **43**(1), 26–40.
- Tutte, W. T. (1954). "A contribution to the theory of chromatic
  polynomials." _Canadian J. Math._ **6**, 80–91.

### Computational complexity

- Jaeger, F.; Vertigan, D. L.; Welsh, D. J. A. (1990). "On the
  computational complexity of the Jones and Tutte polynomials."
  _Math. Proc. Cambridge Philos. Soc._ **108**(1), 35–53.
  _(`#P`-hardness in general)_
- Andrzejak, A. (1998). "An algorithm for the Tutte polynomials of graphs of
  bounded treewidth." _Discrete Mathematics_ **190**(1-3), 39–54.
- Noble, S. D. (1998). "Evaluating the Tutte polynomial for graphs of
  bounded tree-width." _Combin. Probab. Comput._ **7**(3), 307–321.

### Treewidth and tree decompositions

- Robertson, N.; Seymour, P. D. (1984). "Graph minors III: Planar tree-width."
  _J. Combin. Theory Ser. B_ **36**(1), 49–64.
- Bodlaender, H. L. (1996). "A linear-time algorithm for finding
  tree-decompositions of small treewidth." _SIAM J. Comput._ **25**(6),
  1305–1317.

### Cographs and clique-width

- Corneil, D. G.; Lerchs, H.; Stewart Burlingham, L. (1981). "Complement
  reducible graphs." _Discrete Applied Mathematics_ **3**(3), 163–174.
  _(`P_4`-free characterization of cographs)_
- Courcelle, B.; Olariu, S. (2000). "Upper bounds to the clique-width of
  graphs." _Discrete Applied Mathematics_ **101**(1–3), 77–114.
- Giménez, O.; Hliněný, P.; Noy, M. (2006). "Computing the Tutte polynomial
  on graphs of bounded clique-width." _SIAM J. Discrete Math._ **20**(4),
  932–946. _(cotree DP foundation)_

### k-vertex sums and matroid theory

- Brylawski, T. H. (1971). "A combinatorial model for series-parallel
  networks." _Trans. Amer. Math. Soc._ **154**, 1–22. _(2-sum identity)_
- Oxley, J. G. (2011). _Matroid Theory_ (2nd ed.). Oxford University Press.
  _(comprehensive reference; chapter 11 covers k-sum identities)_
- Bonin, J.; de Mier, A. (2004). "T-uniqueness of some families of
  k-chordal matroids." _Advances in Applied Mathematics_ **32**, 10–30.
  _(matroid-theoretic k-sum approach; retired in our pipeline)_

### Multivariate Tutte / Potts model

- Sokal, A. D. (2005). "The multivariate Tutte polynomial (alias Potts
  model) for graphs and matroids." In: _Surveys in Combinatorics, 2005_.
  _(comprehensive multivariate / Potts survey; `arxiv:math/0503607`)_

### Specialized algorithms / recent results

- Sun Tzu (~3rd century CE); Gauss, C. F. (1801). _Disquisitiones
  Arithmeticae_. _(Chinese Remainder Theorem)_
- Kirchhoff, G. (1847). "Über die Auflösung der Gleichungen, auf welche man
  bei der Untersuchung der linearen Verteilung galvanischer Ströme geführt
  wird." _Annalen der Physik_. _(matrix-tree theorem)_
- Tarjan, R. (1972). "Depth-first search and linear graph algorithms."
  _SIAM J. Comput._ **1**(2), 146–160. _(linear-time cut-vertex detection)_
- Weisfeiler, B.; Leman, A. A. (1968). "A reduction of a graph to a
  canonical form and an algebra arising during this reduction" (Russian).
  _Nauchno-Technicheskaya Informatsia_. _(canonical key foundation)_
- Yardim, B.; Türker, T. (2025). "Linear-time exact Potts on series-parallel
  graphs." `arXiv:2507.22579`. _(exact SP partition function)_
- Blažej, V.; Jana, S.; Ramanujan, M. S. (2025). "Cograph-modular treewidth."
  IPEC 2025. _(structural parameter strictly between treewidth and
  clique-width; potential D-Wave fit)_
