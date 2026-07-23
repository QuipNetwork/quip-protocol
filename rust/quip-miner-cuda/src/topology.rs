//! CSR topology + chromatic color-blocks + int8 quantization for the
//! self-feeding kernels.
//!
//! Mirrors `GPU/sampler_utils.py::build_csr_structure_from_edges` /
//! `build_edge_position_index` / `compute_color_blocks`, but computes a
//! generic greedy coloring instead of the Zephyr-specific linear-index
//! formula (`zephyr_four_color_linear`): the kernel takes `num_colors` as a
//! runtime argument, so any valid coloring (same-color nodes non-adjacent)
//! is correct, and a greedy coloring works for any topology, not just
//! Zephyr.
//!
//! Consensus `h`/`J` are constrained to small integers by protocol design
//! (`DEFAULT_ALLOWED_H = {-1,0,1}`, `DEFAULT_ALLOWED_J = {-1,1}`, milli
//! units; see `shared/quantum_proof_of_work.py`), so the int8 cast the
//! original kernel relies on is lossless for real jobs.

use quip_miner_core::IsingGraph;

/// Chromatic color-block partition of a CSR graph's dense node indices.
///
/// `nodes` is grouped by color; `starts`/`counts` index into it per color.
/// Same-color nodes are pairwise non-adjacent (independent set), which is
/// all the kernel's per-color parallel update requires.
#[derive(Clone, Debug)]
pub struct ColorBlocks {
    pub starts: Vec<i32>,
    pub counts: Vec<i32>,
    pub nodes: Vec<i32>,
    pub num_colors: i32,
}

/// Greedy (Welsh-Powell) coloring of a CSR adjacency: process nodes in
/// degree-descending order, assign the smallest color unused by any
/// already-colored neighbor. Not the Zephyr-optimal 4-coloring, but valid
/// for any graph and typically close to it for sparse Ising topologies.
fn greedy_color(n: usize, row_ptr: &[i32], col_ind: &[i32]) -> ColorBlocks {
    if n == 0 {
        return ColorBlocks {
            starts: Vec::new(),
            counts: Vec::new(),
            nodes: Vec::new(),
            num_colors: 0,
        };
    }
    let degree = |i: usize| (row_ptr[i + 1] - row_ptr[i]) as usize;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(degree(i)));

    let mut color_of = vec![-1i32; n];
    let mut used = vec![false; n]; // reused scratch, cleared per node
    for &node in &order {
        let start = row_ptr[node] as usize;
        let end = row_ptr[node + 1] as usize;
        let mut touched: Vec<usize> = Vec::with_capacity(end - start);
        for &nbr in &col_ind[start..end] {
            let c = color_of[nbr as usize];
            if c >= 0 {
                used[c as usize] = true;
                touched.push(c as usize);
            }
        }
        let mut c = 0usize;
        while c < n && used[c] {
            c += 1;
        }
        color_of[node] = c as i32;
        for t in touched {
            used[t] = false;
        }
    }

    let num_colors = color_of.iter().copied().max().unwrap_or(-1) + 1;
    let mut groups: Vec<Vec<i32>> = vec![Vec::new(); num_colors.max(0) as usize];
    for (i, &c) in color_of.iter().enumerate() {
        groups[c as usize].push(i as i32);
    }
    let mut starts = Vec::with_capacity(groups.len());
    let mut counts = Vec::with_capacity(groups.len());
    let mut nodes = Vec::with_capacity(n);
    let mut cur = 0i32;
    for g in &groups {
        starts.push(cur);
        counts.push(g.len() as i32);
        nodes.extend_from_slice(g);
        cur += g.len() as i32;
    }
    ColorBlocks {
        starts,
        counts,
        nodes,
        num_colors,
    }
}

/// Fixed CSR topology shared by every nonce/slot in a self-feeding session.
///
/// Built once from the first job's graph. Subsequent jobs must supply the
/// exact same `(n, edges)` (checked by the caller via [`IsingGraph`]
/// equality) to reuse it; `edge_pos` gives each edge's two CSR positions in
/// that fixed order, so per-job `J` upload is a direct scatter with no
/// per-job graph traversal.
#[derive(Clone, Debug)]
pub struct SelfFeedingTopology {
    pub n: usize,
    pub nnz: usize,
    pub row_ptr: Vec<i32>,
    pub col_ind: Vec<i32>,
    /// Per-edge `(pos_ij, pos_ji)` into `col_ind`/`j` arrays, parallel to the
    /// establishing graph's `edges` order.
    pub edge_pos: Vec<(u32, u32)>,
    pub colors: ColorBlocks,
}

impl SelfFeedingTopology {
    /// Build CSR + coloring from a graph. `graph.edges` fixes the canonical
    /// edge order used by `edge_pos` (and thus by [`fill_h_j`] for this and
    /// every later job sharing this topology).
    pub fn build(graph: &IsingGraph) -> Self {
        let n = graph.h.len();
        // Per-node list of (neighbor, edge_index, is_forward_half): carries
        // the originating `graph.edges[edge_index]` through the sort so the
        // final CSR position can be written straight into `edge_pos` below,
        // with no post-hoc search for "where did this edge end up".
        let mut adj: Vec<Vec<(usize, usize, bool)>> = vec![Vec::new(); n];
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            adj[u].push((v, k, true));
            if u != v {
                adj[v].push((u, k, false));
            }
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable_by_key(|&(nbr, _, _)| nbr);
        }

        let mut row_ptr = vec![0i32; n + 1];
        let mut col_ind = Vec::new();
        // (0, 0) for an edge with an out-of-range endpoint: never read,
        // since `fill_h_j` skips those edges too (matches the guard above).
        let mut edge_pos = vec![(0u32, 0u32); graph.edges.len()];
        for i in 0..n {
            row_ptr[i] = col_ind.len() as i32;
            for &(nbr, k, is_forward) in &adj[i] {
                let pos = col_ind.len() as u32;
                col_ind.push(nbr as i32);
                if is_forward {
                    edge_pos[k].0 = pos;
                } else {
                    edge_pos[k].1 = pos;
                }
            }
        }
        row_ptr[n] = col_ind.len() as i32;
        let nnz = col_ind.len();

        let colors = greedy_color(n, &row_ptr, &col_ind);

        Self {
            n,
            nnz,
            row_ptr,
            col_ind,
            edge_pos,
            colors,
        }
    }
}

/// Truncating cast to int8, saturating on overflow (Rust's `as` semantics
/// since 1.45). Matches numpy's `dtype=np.int8` cast for the in-range
/// values consensus actually produces (h in {-1,0,1}, J in {-1,1}, milli
/// units); saturates instead of wrapping for out-of-range test fixtures.
fn quantize_i8(v: f64) -> i8 {
    v as i8
}

/// Quantize one job's `h`/`J` into the topology's fixed CSR layout.
///
/// `j_csr` has length `topology.nnz`; `h_i8` has length `topology.n`.
/// Positions not touched by any edge stay `0` (matches `j_csr` being
/// allocated/cleared before this call).
pub fn fill_h_j(topology: &SelfFeedingTopology, graph: &IsingGraph) -> (Vec<i8>, Vec<i8>) {
    let mut j_csr = vec![0i8; topology.nnz];
    for (k, &(pos_ij, pos_ji)) in topology.edge_pos.iter().enumerate() {
        let (u, v) = graph.edges[k];
        if u >= topology.n || v >= topology.n {
            continue;
        }
        let val = quantize_i8(graph.j.get(k).copied().unwrap_or(0.0));
        j_csr[pos_ij as usize] = val;
        if u != v {
            j_csr[pos_ji as usize] = val;
        }
    }
    let h_i8: Vec<i8> = graph.h.iter().map(|&v| quantize_i8(v)).collect();
    (j_csr, h_i8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> IsingGraph {
        // Small ring: 0-1-2-3-0, unit J, ternary h.
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    #[test]
    fn csr_shape_and_symmetry() {
        let t = SelfFeedingTopology::build(&g());
        assert_eq!(t.n, 4);
        assert_eq!(t.nnz, 8); // 4 edges * 2 directed halves
        assert_eq!(t.row_ptr, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn coloring_is_proper() {
        let t = SelfFeedingTopology::build(&g());
        // Every node gets exactly one color; adjacent nodes differ.
        let mut color_of = vec![-1i32; t.n];
        for (c, (&start, &count)) in t.colors.starts.iter().zip(&t.colors.counts).enumerate() {
            for i in 0..count {
                let node = t.colors.nodes[(start + i) as usize] as usize;
                assert_eq!(color_of[node], -1, "node colored twice");
                color_of[node] = c as i32;
            }
        }
        assert!(color_of.iter().all(|&c| c >= 0), "every node colored");
        for i in 0..t.n {
            let s = t.row_ptr[i] as usize;
            let e = t.row_ptr[i + 1] as usize;
            for &nbr in &t.col_ind[s..e] {
                assert_ne!(
                    color_of[i],
                    color_of[nbr as usize],
                    "adjacent nodes {i} and {nbr} share a color"
                );
            }
        }
    }

    #[test]
    fn quantization_is_lossless_for_consensus_range() {
        let t = SelfFeedingTopology::build(&g());
        let (j, h) = fill_h_j(&t, &g());
        assert_eq!(h, vec![1i8, -1, 0, 1]);
        // Each edge's J appears at both directed CSR positions.
        assert_eq!(j.iter().filter(|&&v| v != 0).count(), 8);
    }

    #[test]
    fn empty_graph_has_no_colors() {
        let t = SelfFeedingTopology::build(&IsingGraph::new(vec![], vec![], vec![]));
        assert_eq!(t.n, 0);
        assert_eq!(t.colors.num_colors, 0);
    }
}
