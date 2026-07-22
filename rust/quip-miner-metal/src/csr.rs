//! CSR (compressed sparse row) builder for undirected Ising graphs.
//!
//! Couplings are stored once per directed half-edge so local-field walks are
//! O(degree). Value dtype is f32 for kernel dynamics; the host keeps f64
//! copies for consensus scoring.

/// Dense Ising problem with CSR adjacency for GPU upload.
#[derive(Clone, Debug)]
pub struct IsingGraph {
    pub h: Vec<f64>,
    pub j: Vec<f64>,
    pub edges: Vec<(usize, usize)>,
    /// CSR row pointers, length `N + 1`.
    pub row_ptr: Vec<i32>,
    /// CSR column indices, length `nnz`.
    pub col_ind: Vec<i32>,
    /// CSR coupling values (symmetric half-edges), length `nnz`.
    pub j_csr: Vec<f32>,
    /// Linear biases as f32 for kernel upload.
    pub h_f32: Vec<f32>,
}

impl IsingGraph {
    /// Build adjacency from flat h / j / edge lists.
    ///
    /// Edges whose endpoints are out of range for `h.len()` are skipped (same
    /// defensive posture as `energy_milli`). Couplings shorter than edges are
    /// treated as 0 for the missing entries.
    pub fn new(h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> Self {
        let n = h.len();
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (k, &(u, v)) in edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            let coup = j.get(k).copied().unwrap_or(0.0) as f32;
            adj[u].push((v, coup));
            if u != v {
                adj[v].push((u, coup));
            }
        }
        // Deterministic neighbor order (matches Python GPU CSR builder).
        for nbrs in &mut adj {
            nbrs.sort_by_key(|&(idx, _)| idx);
        }

        let mut row_ptr = vec![0i32; n + 1];
        let mut col_ind = Vec::new();
        let mut j_csr = Vec::new();
        for i in 0..n {
            for &(nbr, coup) in &adj[i] {
                col_ind.push(nbr as i32);
                j_csr.push(coup);
            }
            row_ptr[i + 1] = col_ind.len() as i32;
        }

        let h_f32: Vec<f32> = h.iter().map(|&v| v as f32).collect();
        Self {
            h,
            j,
            edges,
            row_ptr,
            col_ind,
            j_csr,
            h_f32,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.h.len()
    }

    pub fn nnz(&self) -> usize {
        self.col_ind.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_symmetric_two_node() {
        let g = IsingGraph::new(vec![1.0, -1.0], vec![0.5], vec![(0, 1)]);
        assert_eq!(g.row_ptr, vec![0, 1, 2]);
        assert_eq!(g.col_ind, vec![1, 0]);
        assert_eq!(g.j_csr, vec![0.5, 0.5]);
        assert_eq!(g.nnz(), 2);
    }

    #[test]
    fn empty_graph() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        assert_eq!(g.row_ptr, vec![0]);
        assert_eq!(g.nnz(), 0);
    }
}
