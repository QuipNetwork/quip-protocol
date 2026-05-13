"""C extension for partition operations on the path/grid DP hot path.

Hot path: `precompute_M_table` calls `join_partitions` 14M+ times for
Cm3-class graphs (52% of total time per cProfile). Each call is union-find
over a small universe (≤24 verts).

Pattern mirrors `tutte/_polynomial_c.py`: lazy auto-compile on first import.
Pure-Python fallback if cffi compile fails.

Encoding: a partition is encoded as a flat int array
  [block_count, block_1_size, b1_v1, b1_v2, ..., block_2_size, b2_v1, ...]
The C function takes encoded p1, p2, and a universe array (the set of
positions); returns the joined partition in the same encoding.

Positions are relabeled to 0..n_universe-1 in Python before the C call
to avoid hash-table needs in C. Output positions get re-mapped back.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cffi

ffi = cffi.FFI()

ffi.cdef(r"""
    /* Joins two encoded partitions over a universe of size n_universe.
       Positions in p1, p2 must be in [0, n_universe).
       p1_data: [block_count, b1_size, b1_v1, ..., b2_size, ...]
       p2_data: same encoding.
       out_data: pre-allocated output buffer, capacity ints.
       Returns the number of ints written to out_data, or -1 on overflow.

       Output is canonically sorted (blocks sorted by min element;
       within each block, vertices sorted ascending). */
    int join_partitions_c(
        const int* p1_data, int p1_len,
        const int* p2_data, int p2_len,
        int n_universe,
        int* out_data, int out_capacity);

    /* Batched inner loop for precompute_M_table.

       For ONE state partition (P_state_ext), processes ALL junction
       partitions in batch. For each junction:
         1. Compute delta on shared boundary (positions [0, n_shared)).
         2. If delta < 0: skip (output_deltas[i] = -1).
         3. Else: join over full universe, restrict to out_boundary,
            compute per-cell canonical key.
         4. Output: delta + out canonical key (encoded).

       All position arrays use indices in [0, n_universe). Caller has
       already relabeled. The cell_groups array is also in [0, n_universe).

       boundaries_data: flat array containing all boundary subsets,
         indexed by *_start, *_end positions:
           shared = [0, n_shared)
           extra = [n_shared, n_shared + n_extra)
           state_extra = [n_shared + n_extra, n_shared + n_extra + n_state_extra)
           out = caller computes from above + keep_shared flag, passes separately.

       Returns: total ints written to out_buf (= sum of per-junction output
       segments + one int per junction for the delta), or -1 on overflow.

       Output layout per junction j (in out_buf):
         [delta_j, n_canonical_ints, canonical_data...]
       If delta < 0: just [-1, 0]. */
    int batched_inner_iterations_c(
        /* State partition (extended with extra_boundary as singletons) */
        const int* p_state_ext, int p_state_ext_len,
        /* State partition restricted to shared boundary (precomputed) */
        const int* p_state_S, int p_state_S_len,
        /* All junction partitions in this batch (encoded + extended) */
        const int* junc_parts_ext_data, const int* junc_parts_ext_offsets,
        const int* junc_parts_ext_lens,
        const int* junc_parts_S_data, const int* junc_parts_S_offsets,
        const int* junc_parts_S_lens,
        int n_junc,
        /* Universe parameters */
        int n_universe, int n_shared,
        /* Out boundary indices in [0, n_universe) */
        const int* out_boundary_idx, int n_out_boundary,
        /* Cell anchor groups for canonical key.
           Encoded: [n_cells, c0_size, c0_v0, ..., c1_size, c1_v0, ...] */
        const int* cell_groups_data, int cell_groups_len, int n_cells,
        /* Output buffer: one segment per junction (variable length).
           Returns total ints written, or -1 on capacity exceeded. */
        int* out_buf, int out_capacity,
        int* out_offsets);
""")

ffi.set_source("_tutte_partition_cffi", r"""
    #include <stdlib.h>
    #include <string.h>

    /* Path-compressing find. */
    static int uf_find(int* parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /* Union by min-rep (matches Python convention). */
    static void uf_union(int* parent, int a, int b) {
        int ra = uf_find(parent, a);
        int rb = uf_find(parent, b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
    }

    /* Apply unions encoded in `data` (one partition's blocks) to parent. */
    static void apply_blocks(int* parent, const int* data, int n) {
        if (n <= 0) return;
        int block_count = data[0];
        int idx = 1;
        for (int b = 0; b < block_count && idx < n; b++) {
            int bsize = data[idx++];
            if (bsize >= 1 && idx < n) {
                int rep = data[idx];
                for (int i = 1; i < bsize && idx + i < n; i++) {
                    uf_union(parent, rep, data[idx + i]);
                }
            }
            idx += bsize;
        }
    }

    /* Comparator for sorting block-tuples canonically. We sort blocks by
       lexicographic order; each block's vertices are sorted ascending
       beforehand, so block-ordering comparison reduces to comparing
       (size, then per-position values). For our tiny block counts (~24),
       a stable insertion sort is fastest. */

    int join_partitions_c(
        const int* p1_data, int p1_len,
        const int* p2_data, int p2_len,
        int n_universe,
        int* out_data, int out_capacity)
    {
        if (n_universe <= 0) {
            if (out_capacity < 1) return -1;
            out_data[0] = 0;
            return 1;
        }

        /* Stack-allocated parent array (universe ≤ 64 should cover all
           practical cases — Cm3 has 12-24 verts max). */
        int parent[256];
        if (n_universe > 256) return -1;
        for (int i = 0; i < n_universe; i++) parent[i] = i;

        /* Apply unions from both partitions. */
        apply_blocks(parent, p1_data, p1_len);
        apply_blocks(parent, p2_data, p2_len);

        /* Group by component. block_of[v] = index of v's block in output. */
        int block_of[256];
        int blocks[256][256];   /* block contents (overkill but safe) */
        int block_sizes[256];
        int n_blocks = 0;
        int rep_to_block[256];
        for (int i = 0; i < n_universe; i++) rep_to_block[i] = -1;

        for (int v = 0; v < n_universe; v++) {
            int r = uf_find(parent, v);
            if (rep_to_block[r] < 0) {
                rep_to_block[r] = n_blocks;
                block_sizes[n_blocks] = 0;
                n_blocks++;
            }
            int bidx = rep_to_block[r];
            blocks[bidx][block_sizes[bidx]++] = v;
        }

        /* Each block is already sorted (we iterated v in 0..n-1).
           Now sort blocks by (first vertex), which is the canonical order. */
        /* Simple insertion sort on block indices by blocks[idx][0]. */
        int order[256];
        for (int i = 0; i < n_blocks; i++) order[i] = i;
        for (int i = 1; i < n_blocks; i++) {
            int key = order[i];
            int j = i - 1;
            while (j >= 0 && blocks[order[j]][0] > blocks[key][0]) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        /* Encode output: [n_blocks, b1_size, b1_v1, ..., b2_size, ...]. */
        int out_idx = 0;
        if (out_capacity < 1) return -1;
        out_data[out_idx++] = n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            int bidx = order[b];
            int bsize = block_sizes[bidx];
            if (out_idx + 1 + bsize > out_capacity) return -1;
            out_data[out_idx++] = bsize;
            for (int i = 0; i < bsize; i++) {
                out_data[out_idx++] = blocks[bidx][i];
            }
        }
        return out_idx;
    }

    /* === Batched inner loop for precompute_M_table === */

    /* Compute delta(P_state_S, P_junc_S) on shared boundary.
       Both partitions use indices in [0, n_shared).
       Returns delta = nblocks(JOIN) + n_shared - nblocks(P_state_S) - nblocks(P_junc_S). */
    static int compute_delta(
        const int* p_state_S, int p_state_S_len,
        const int* p_junc_S, int p_junc_S_len,
        int n_shared)
    {
        if (n_shared == 0) {
            return 0 - 0 - 0;  /* nblocks(empty) = 0 */
        }
        int parent[256];
        for (int i = 0; i < n_shared; i++) parent[i] = i;
        apply_blocks(parent, p_state_S, p_state_S_len);
        apply_blocks(parent, p_junc_S, p_junc_S_len);
        /* Count distinct components */
        int seen[256] = {0};
        int n_join_blocks = 0;
        for (int v = 0; v < n_shared; v++) {
            int r = uf_find(parent, v);
            if (!seen[r]) { seen[r] = 1; n_join_blocks++; }
        }
        int n_state_blocks = (p_state_S_len > 0) ? p_state_S[0] : 0;
        int n_junc_blocks = (p_junc_S_len > 0) ? p_junc_S[0] : 0;
        return n_join_blocks + n_shared - n_state_blocks - n_junc_blocks;
    }

    /* Compute joint partition on full universe (size n_universe).
       Then restrict to out_boundary_idx, then compute per-cell canonical key.
       Output written as: [n_canonical_blocks, shape_0_per_cell..., shape_1_per_cell...]
       where each shape is n_cells+1 ints (last = "outside" count = 0 for valid cases).
       Returns: number of ints written. */
    static int join_restrict_canonical(
        const int* p_state_ext, int p_state_ext_len,
        const int* p_junc_ext, int p_junc_ext_len,
        int n_universe,
        const int* out_boundary_idx, int n_out_boundary,
        const int* cell_groups_data, int n_cells,
        int* out_canonical_buf, int buf_capacity)
    {
        if (n_out_boundary == 0) {
            /* Empty out boundary: canonical key is empty tuple */
            if (buf_capacity < 1) return -1;
            out_canonical_buf[0] = 0;  /* n_blocks = 0 */
            return 1;
        }
        if (n_universe > 256) return -1;

        int parent[256];
        for (int i = 0; i < n_universe; i++) parent[i] = i;
        apply_blocks(parent, p_state_ext, p_state_ext_len);
        apply_blocks(parent, p_junc_ext, p_junc_ext_len);

        /* Build pos_to_cell map: for each index in [0, n_universe),
           which cell does it belong to? -1 = outside. */
        int pos_to_cell[256];
        for (int i = 0; i < n_universe; i++) pos_to_cell[i] = -1;
        int idx = 0;
        if (cell_groups_data[idx++] != n_cells) return -1;  /* sanity */
        for (int c = 0; c < n_cells; c++) {
            int csize = cell_groups_data[idx++];
            for (int i = 0; i < csize; i++) {
                int v = cell_groups_data[idx++];
                if (v >= 0 && v < n_universe) pos_to_cell[v] = c;
            }
        }

        /* Build out_in_universe lookup: is index v in out boundary? */
        int in_out[256] = {0};
        for (int i = 0; i < n_out_boundary; i++) in_out[out_boundary_idx[i]] = 1;

        /* For each component (root), compute its shape over cells +
           outside (n_cells+1 slots), but only counting positions in
           out_boundary. */
        int comp_shape[256][16];  /* n_cells+1 ≤ 16 in practice */
        int comp_seen[256] = {0};
        int n_comps_in_out = 0;
        int comp_root_to_idx[256];
        for (int i = 0; i < n_universe; i++) comp_root_to_idx[i] = -1;
        if (n_cells + 1 > 16) return -1;

        for (int v = 0; v < n_universe; v++) {
            if (!in_out[v]) continue;
            int r = uf_find(parent, v);
            int ci = pos_to_cell[v] >= 0 ? pos_to_cell[v] : n_cells;
            if (!comp_seen[r]) {
                comp_seen[r] = 1;
                comp_root_to_idx[r] = n_comps_in_out;
                for (int j = 0; j < n_cells + 1; j++) comp_shape[n_comps_in_out][j] = 0;
                n_comps_in_out++;
            }
            comp_shape[comp_root_to_idx[r]][ci]++;
        }

        /* Build sorted multiset of shape tuples. Each tuple has n_cells+1 ints.
           Sort blocks lexicographically. Use insertion sort. */
        int order[256];
        for (int i = 0; i < n_comps_in_out; i++) order[i] = i;
        for (int i = 1; i < n_comps_in_out; i++) {
            int key = order[i];
            int j = i - 1;
            while (j >= 0) {
                /* Compare comp_shape[order[j]] vs comp_shape[key] lex. */
                int cmp = 0;
                for (int k = 0; k < n_cells + 1; k++) {
                    if (comp_shape[order[j]][k] < comp_shape[key][k]) { cmp = -1; break; }
                    if (comp_shape[order[j]][k] > comp_shape[key][k]) { cmp = 1; break; }
                }
                if (cmp <= 0) break;
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        /* Encode: [n_blocks, shape_0_data..., shape_1_data..., ...] */
        int needed = 1 + n_comps_in_out * (n_cells + 1);
        if (buf_capacity < needed) return -1;
        int out_idx = 0;
        out_canonical_buf[out_idx++] = n_comps_in_out;
        for (int b = 0; b < n_comps_in_out; b++) {
            int bidx = order[b];
            for (int k = 0; k < n_cells + 1; k++) {
                out_canonical_buf[out_idx++] = comp_shape[bidx][k];
            }
        }
        return out_idx;
    }

    int batched_inner_iterations_c(
        const int* p_state_ext, int p_state_ext_len,
        const int* p_state_S, int p_state_S_len,
        const int* junc_parts_ext_data, const int* junc_parts_ext_offsets,
        const int* junc_parts_ext_lens,
        const int* junc_parts_S_data, const int* junc_parts_S_offsets,
        const int* junc_parts_S_lens,
        int n_junc,
        int n_universe, int n_shared,
        const int* out_boundary_idx, int n_out_boundary,
        const int* cell_groups_data, int cell_groups_len, int n_cells,
        int* out_buf, int out_capacity,
        int* out_offsets)
    {
        int total_written = 0;
        for (int j = 0; j < n_junc; j++) {
            out_offsets[j] = total_written;
            const int* p_junc_S = junc_parts_S_data + junc_parts_S_offsets[j];
            int p_junc_S_len = junc_parts_S_lens[j];
            int delta = compute_delta(
                p_state_S, p_state_S_len,
                p_junc_S, p_junc_S_len,
                n_shared
            );
            /* Write delta */
            if (total_written + 1 > out_capacity) return -1;
            out_buf[total_written++] = delta;
            if (delta < 0) {
                /* Skip canonical computation; write 0 for n_canonical_ints */
                if (total_written + 1 > out_capacity) return -1;
                out_buf[total_written++] = 0;
                continue;
            }
            const int* p_junc_ext = junc_parts_ext_data + junc_parts_ext_offsets[j];
            int p_junc_ext_len = junc_parts_ext_lens[j];

            /* Reserve space for canonical_size + canonical_data */
            if (total_written + 1 > out_capacity) return -1;
            int* canonical_size_slot = &out_buf[total_written++];
            int n_can = join_restrict_canonical(
                p_state_ext, p_state_ext_len,
                p_junc_ext, p_junc_ext_len,
                n_universe,
                out_boundary_idx, n_out_boundary,
                cell_groups_data, n_cells,
                out_buf + total_written, out_capacity - total_written
            );
            if (n_can < 0) return -1;
            *canonical_size_slot = n_can;
            total_written += n_can;
        }
        out_offsets[n_junc] = total_written;
        return total_written;
    }
""", extra_compile_args=["-O3"])


_lib = None
_lib_lock = threading.Lock()
_ffi = None


def _get_lib():
    global _lib, _ffi
    if _lib is not None:
        return _lib, _ffi
    with _lib_lock:
        if _lib is not None:
            return _lib, _ffi
        try:
            from _tutte_partition_cffi import ffi as cffi_ffi
            from _tutte_partition_cffi import lib
            _lib = lib
            _ffi = cffi_ffi
            return _lib, _ffi
        except ImportError:
            pass
        import sys
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="tutte_partition_c_")
        ffi.compile(tmpdir=tmpdir)
        sys.path.insert(0, tmpdir)
        from _tutte_partition_cffi import ffi as cffi_ffi
        from _tutte_partition_cffi import lib
        _lib = lib
        _ffi = cffi_ffi
        return _lib, _ffi


def _encode_partition(P: Tuple[Tuple[int, ...], ...], pos_to_idx: dict) -> List[int]:
    """Encode partition P as flat int array using pos_to_idx mapping."""
    out = [len(P)]
    for block in P:
        out.append(len(block))
        for v in block:
            out.append(pos_to_idx[v])
    return out


def _decode_partition(data, n: int, idx_to_pos: List[int]) -> Tuple[Tuple[int, ...], ...]:
    """Decode flat int array (cffi int*, length n) into partition tuple."""
    block_count = data[0]
    blocks = []
    idx = 1
    for _ in range(block_count):
        bsize = data[idx]
        idx += 1
        block = tuple(idx_to_pos[data[idx + i]] for i in range(bsize))
        blocks.append(block)
        idx += bsize
    return tuple(blocks)


from .rooted_tutte import restrict_partition as _restrict_partition_py


def _encode_partition_to_idx(P, pos_to_idx):
    """Encode partition with positions already known to be in pos_to_idx."""
    out = [len(P)]
    for block in P:
        out.append(len(block))
        for v in block:
            out.append(pos_to_idx[v])
    return out


def precompute_M_batched_inner_c(
    state_orbit_partitions,           # dict canonical → [rep partition]
    junc_data_per_orbit,              # dict junc_canonical → list of (P_junc, P_junc_S, P_junc_ext)
    state_extra_boundary,             # list[int]
    extra_boundary,                   # list[int]
    shared_boundary,                  # list[int]
    out_boundary,                     # list[int]
    out_cell_anchor_groups,           # list[list[int]] — for canonical key
    n_state_per_orbit,                # dict canonical → n_state (analytical)
    xy_powers_dict,                   # list[dict[(xpow, ypow), int]] indexed by d
):
    """C-batched version of precompute_M_table's inner loop.

    Returns: dict (O_state, O_junc, O_out_canonical_tuple) → coeff dict.
    Returns None if C extension unavailable, sizes exceed limits, or any error.

    For each outer state orbit, batches ALL junction partitions in one C call.
    For Cm3 row composition: 6608 outer × 6608 batched per call → 6608 cffi calls
    + minimal Python overhead per pair. Estimated 10-50x speedup.
    """
    # Build universe = state_extra + shared + extra (matches precompute_M_table).
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None  # exceeds C buffer
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {p: i for i, p in enumerate(full_universe)}
    # Separate mapping for shared_boundary alone — used to encode P_state_S
    # and P_junc_S whose positions live in [0, n_shared) per C's expectation.
    shared_pos_to_idx = {p: i for i, p in enumerate(shared_boundary)}

    # out_boundary indices in [0, n_universe). Some out positions might not be
    # in full_universe (if keep_shared and shared positions are in out but not
    # in extra). For correctness, all out positions must be in full_universe.
    try:
        out_boundary_idx = [pos_to_idx[p] for p in out_boundary]
    except KeyError:
        return None

    # Encode cell_anchor_groups using pos_to_idx. Positions outside universe
    # get index = -1 (treated as "outside" in C).
    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        # Filter to positions in universe.
        in_universe = [pos_to_idx[p] for p in cell_positions if p in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    # Pre-encode all junction partitions (once across all outer iterations).
    # Build flat array + offsets for each junction's P_junc_S and P_junc_ext.
    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    # Flatten: total junction partitions = sum over orbits of len(per_junc)
    junc_flat_metadata = []  # list of (orbit_idx, P_junc_idx_in_orbit)
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_flat_metadata.append((orbit_idx, len(junc_flat_metadata)))
            # Encode P_junc_S using shared_pos_to_idx — positions in
            # shared boundary mapped to [0, n_shared).
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            # Encode P_junc_ext (positions in full_universe)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_flat_metadata)
    if n_junc_total == 0:
        return {}

    # Allocate cffi arrays for junction data.
    junc_S_arr = _ffi.new("int[]", junc_S_flat)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    # Output buffer: per-junction segment is [delta, n_canonical, canonical_data].
    # Worst case: each canonical has n_universe blocks × (n_cells+1) ints + 2.
    per_junc_max = 2 + 1 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    # Process each outer state.
    M_dict = {}

    for O_state, ps_list in state_orbit_partitions.items():
        rep_state = ps_list[0]
        n_state = n_state_per_orbit.get(O_state, len(ps_list))

        # Build P_state_S (restrict rep_state to shared_boundary).
        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        # Build P_state_ext = rep_state + singletons for extra_boundary.
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            # P_state_S uses shared_pos_to_idx (in [0, n_shared)).
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        n_written = lib.batched_inner_iterations_c(
            p_state_ext_arr, len(p_state_ext_data),
            p_state_S_arr, len(p_state_S_data),
            junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
            junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
            n_junc_total,
            n_universe, n_shared,
            out_boundary_arr, n_out_boundary,
            cell_groups_arr, len(cell_groups_data), n_cells,
            out_buf, out_capacity, out_offsets,
        )
        if n_written < 0:
            return None  # capacity exceeded; fall back

        # Decode per-junction results and aggregate. Use buffer protocol
        # to bulk-read cffi int[] into a Python list — avoids 100M+
        # individual cffi attribute accesses.
        out_buf_list = _ffi.unpack(out_buf, n_written) if n_written > 0 else []
        out_offsets_list = _ffi.unpack(out_offsets, n_junc_total + 1)
        shape_size = n_cells + 1
        for j_global in range(n_junc_total):
            orbit_idx, _ = junc_flat_metadata[j_global]
            O_junc = junc_orbit_canonicals[orbit_idx]
            seg_start = out_offsets_list[j_global]
            delta = out_buf_list[seg_start]
            if delta < 0:
                continue
            n_can = out_buf_list[seg_start + 1]
            if n_can == 0:
                O_out = ()
            else:
                n_blocks = out_buf_list[seg_start + 2]
                base0 = seg_start + 3
                # Slice + chunk into block tuples in one Python pass.
                flat = out_buf_list[base0:base0 + n_blocks * shape_size]
                blocks = [tuple(flat[i * shape_size:(i + 1) * shape_size])
                          for i in range(n_blocks)]
                O_out = tuple(sorted(blocks))

            target_key = (O_state, O_junc, O_out)
            if target_key not in M_dict:
                M_dict[target_key] = {}
            target = M_dict[target_key]
            xy_pow = xy_powers_dict[delta]
            for k, v in xy_pow.items():
                target[k] = target.get(k, 0) + v * n_state

    return M_dict


def precompute_M_batched_inner_c_mod(
    state_orbit_partitions,           # dict canonical → [rep partition]
    junc_data_per_orbit,              # dict junc_canonical → list of (P_junc, P_junc_S, P_junc_ext)
    state_extra_boundary,             # list[int]
    extra_boundary,                   # list[int]
    shared_boundary,                  # list[int]
    out_boundary,                     # list[int]
    out_cell_anchor_groups,           # list[list[int]] — for canonical key
    n_state_per_orbit,                # dict canonical → n_state (analytical)
    xy_pow_mod,                       # list[int] — ((x-1)(y-1))^d mod p for d=0..max_d
    p,                                # int — the prime modulus
):
    """Modular variant of `precompute_M_batched_inner_c` (Phase 13.E / Round 14).

    Returns: dict (O_state, O_junc, O_out_canonical_tuple) → int mod p.
    Returns None if C extension unavailable, sizes exceed limits, or any error.

    Reuses `batched_inner_iterations_c` for the structural work
    (delta / join / restrict / per-cell canonical key in C). Only the
    final Python aggregation changes: instead of accumulating polynomial
    coefficient dicts per (O_state, O_junc, O_out) bucket, we accumulate
    a single int mod p per bucket. Removes the inner `for k, v in
    xy_pow.items()` loop entirely — one modular mul-add per (state, junc)
    pair.

    Mirrors the polynomial wrapper's behavior exactly except for arithmetic.
    """
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {pos: i for i, pos in enumerate(full_universe)}
    shared_pos_to_idx = {pos: i for i, pos in enumerate(shared_boundary)}

    try:
        out_boundary_idx = [pos_to_idx[pos] for pos in out_boundary]
    except KeyError:
        return None

    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        in_universe = [pos_to_idx[pos] for pos in cell_positions if pos in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    junc_flat_metadata = []
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_flat_metadata.append((orbit_idx, len(junc_flat_metadata)))
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_flat_metadata)
    if n_junc_total == 0:
        return {}

    junc_S_arr = _ffi.new("int[]", junc_S_flat)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    per_junc_max = 2 + 1 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    M_int = {}

    for O_state, ps_list in state_orbit_partitions.items():
        rep_state = ps_list[0]
        n_state = n_state_per_orbit.get(O_state, len(ps_list))
        n_state_mod = n_state % p
        if n_state_mod == 0:
            continue

        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        n_written = lib.batched_inner_iterations_c(
            p_state_ext_arr, len(p_state_ext_data),
            p_state_S_arr, len(p_state_S_data),
            junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
            junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
            n_junc_total,
            n_universe, n_shared,
            out_boundary_arr, n_out_boundary,
            cell_groups_arr, len(cell_groups_data), n_cells,
            out_buf, out_capacity, out_offsets,
        )
        if n_written < 0:
            return None

        out_buf_list = _ffi.unpack(out_buf, n_written) if n_written > 0 else []
        out_offsets_list = _ffi.unpack(out_offsets, n_junc_total + 1)
        shape_size = n_cells + 1
        for j_global in range(n_junc_total):
            orbit_idx, _ = junc_flat_metadata[j_global]
            O_junc = junc_orbit_canonicals[orbit_idx]
            seg_start = out_offsets_list[j_global]
            delta = out_buf_list[seg_start]
            if delta < 0:
                continue
            n_can = out_buf_list[seg_start + 1]
            if n_can == 0:
                O_out = ()
            else:
                n_blocks = out_buf_list[seg_start + 2]
                base0 = seg_start + 3
                flat = out_buf_list[base0:base0 + n_blocks * shape_size]
                blocks = [tuple(flat[i * shape_size:(i + 1) * shape_size])
                          for i in range(n_blocks)]
                O_out = tuple(sorted(blocks))

            target_key = (O_state, O_junc, O_out)
            add = n_state_mod * xy_pow_mod[delta] % p
            M_int[target_key] = (M_int.get(target_key, 0) + add) % p

    return M_int


def precompute_and_convolve_c_mod(
    state_orbit_partitions,           # dict canonical → [rep partition]
    junc_data_per_orbit,              # dict junc_canonical → list of (P_junc, P_junc_S, P_junc_ext)
    state_extra_boundary,             # list[int]
    extra_boundary,                   # list[int]
    shared_boundary,                  # list[int]
    out_boundary,                     # list[int]
    out_cell_anchor_groups,           # list[list[int]] — for canonical key
    n_state_per_orbit,                # dict canonical → n_state (analytical)
    state_orbit_T_mod,                # dict canonical → int mod p (state value)
    junction_orbit_T_mod,             # dict canonical → int mod p (junction value)
    xy_pow_mod,                       # list[int]
    p,                                # int
):
    """Single-pass C-ext: directly accumulate into out_mod[O_out] (Round 16).

    Combines `precompute_M_batched_inner_c_mod` (build M_int per chunk) +
    streaming-wrapper convolve (M_int × state_T × junc_T → out_mod) into
    one Python loop. Eliminates:
    - 3-tuple `(O_state, O_junc, O_out)` keys (replaced with 1-tuple O_out)
    - Intermediate M_int dict with millions of entries (replaced with a
      few-thousand-entry out_mod)
    - Two dict ops per pair (replaced with one)

    For Cm₃ 2b (row composition) where per-pair Python dict ops dominate
    wall-clock, this is the primary speedup over Round 14/15.

    Returns: dict O_out_canonical_tuple → int mod p (the per-chunk
    contribution to the final out_mod). Caller accumulates across chunks.
    """
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    n_universe = len(full_universe)
    if n_universe > 256:
        return None
    n_shared = len(shared_boundary)
    n_out_boundary = len(out_boundary)

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {pos: i for i, pos in enumerate(full_universe)}
    shared_pos_to_idx = {pos: i for i, pos in enumerate(shared_boundary)}

    try:
        out_boundary_idx = [pos_to_idx[pos] for pos in out_boundary]
    except KeyError:
        return None

    n_cells = len(out_cell_anchor_groups)
    cell_groups_data = [n_cells]
    for cell_positions in out_cell_anchor_groups:
        in_universe = [pos_to_idx[pos] for pos in cell_positions if pos in pos_to_idx]
        cell_groups_data.append(len(in_universe))
        cell_groups_data.extend(in_universe)

    junc_orbit_canonicals = list(junc_data_per_orbit.keys())
    junc_orbit_partition_lists = [junc_data_per_orbit[O_junc] for O_junc in junc_orbit_canonicals]

    junc_flat_metadata = []  # (orbit_idx, _) — but we also need the orbit canonical to lookup jv.
    junc_jv_mod = []  # per-flattened-junc jv mod p value
    junc_S_flat = []
    junc_S_offsets = []
    junc_S_lens = []
    junc_ext_flat = []
    junc_ext_offsets = []
    junc_ext_lens = []

    for orbit_idx, per_junc_list in enumerate(junc_orbit_partition_lists):
        O_junc = junc_orbit_canonicals[orbit_idx]
        jv = junction_orbit_T_mod.get(O_junc, 0)
        if jv == 0:
            # All members of this orbit contribute 0 — skip enumeration.
            for _ in per_junc_list:
                junc_flat_metadata.append((orbit_idx, 0))
                junc_jv_mod.append(0)
                junc_S_offsets.append(len(junc_S_flat))
                junc_S_lens.append(0)
                junc_ext_offsets.append(len(junc_ext_flat))
                junc_ext_lens.append(0)
            continue
        for P_junc, P_junc_S, P_junc_ext in per_junc_list:
            junc_flat_metadata.append((orbit_idx, 0))
            junc_jv_mod.append(jv)
            try:
                p_S_data = _encode_partition_to_idx(P_junc_S, shared_pos_to_idx)
            except KeyError:
                return None
            junc_S_offsets.append(len(junc_S_flat))
            junc_S_lens.append(len(p_S_data))
            junc_S_flat.extend(p_S_data)
            try:
                p_ext_data = _encode_partition_to_idx(P_junc_ext, pos_to_idx)
            except KeyError:
                return None
            junc_ext_offsets.append(len(junc_ext_flat))
            junc_ext_lens.append(len(p_ext_data))
            junc_ext_flat.extend(p_ext_data)

    n_junc_total = len(junc_flat_metadata)
    if n_junc_total == 0:
        return {}

    junc_S_arr = _ffi.new("int[]", junc_S_flat) if junc_S_flat else _ffi.new("int[]", 1)
    junc_S_off_arr = _ffi.new("int[]", junc_S_offsets)
    junc_S_lens_arr = _ffi.new("int[]", junc_S_lens)
    junc_ext_arr = _ffi.new("int[]", junc_ext_flat) if junc_ext_flat else _ffi.new("int[]", 1)
    junc_ext_off_arr = _ffi.new("int[]", junc_ext_offsets)
    junc_ext_lens_arr = _ffi.new("int[]", junc_ext_lens)

    out_boundary_arr = _ffi.new("int[]", out_boundary_idx) if out_boundary_idx else _ffi.new("int[]", 1)
    cell_groups_arr = _ffi.new("int[]", cell_groups_data)

    per_junc_max = 2 + 1 + n_universe * (n_cells + 1)
    out_capacity = n_junc_total * per_junc_max
    out_buf = _ffi.new("int[]", out_capacity)
    out_offsets = _ffi.new("int[]", n_junc_total + 1)

    out_mod = {}

    for O_state, ps_list in state_orbit_partitions.items():
        sv = state_orbit_T_mod.get(O_state, 0)
        if sv == 0:
            continue
        n_state = n_state_per_orbit.get(O_state, len(ps_list))
        n_state_mod = n_state % p
        if n_state_mod == 0:
            continue
        sv_n_state = sv * n_state_mod % p
        if sv_n_state == 0:
            continue

        rep_state = ps_list[0]
        P_state_S = _restrict_partition_py(rep_state, shared_boundary)
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))

        try:
            p_state_ext_data = _encode_partition_to_idx(P_state_ext, pos_to_idx)
            p_state_S_data = _encode_partition_to_idx(P_state_S, shared_pos_to_idx)
        except KeyError:
            return None

        p_state_ext_arr = _ffi.new("int[]", p_state_ext_data)
        p_state_S_arr = _ffi.new("int[]", p_state_S_data)

        n_written = lib.batched_inner_iterations_c(
            p_state_ext_arr, len(p_state_ext_data),
            p_state_S_arr, len(p_state_S_data),
            junc_ext_arr, junc_ext_off_arr, junc_ext_lens_arr,
            junc_S_arr, junc_S_off_arr, junc_S_lens_arr,
            n_junc_total,
            n_universe, n_shared,
            out_boundary_arr, n_out_boundary,
            cell_groups_arr, len(cell_groups_data), n_cells,
            out_buf, out_capacity, out_offsets,
        )
        if n_written < 0:
            return None

        out_buf_list = _ffi.unpack(out_buf, n_written) if n_written > 0 else []
        out_offsets_list = _ffi.unpack(out_offsets, n_junc_total + 1)
        shape_size = n_cells + 1
        # Use bytes as O_out key — much faster construction + hash than
        # nested tuple-of-tuples. The C output is already canonically sorted,
        # so slicing the int run directly yields a stable key.
        for j_global in range(n_junc_total):
            jv = junc_jv_mod[j_global]
            if jv == 0:
                continue
            seg_start = out_offsets_list[j_global]
            delta = out_buf_list[seg_start]
            if delta < 0:
                continue
            n_can = out_buf_list[seg_start + 1]
            if n_can == 0:
                O_out_bytes = b""
            else:
                n_blocks = out_buf_list[seg_start + 2]
                base0 = seg_start + 3
                # bytes(memoryview): O(n) but no per-tuple Python alloc.
                # Each int is 4 bytes (ffi int = C int).
                O_out_bytes = bytes(_ffi.buffer(
                    out_buf + base0, n_blocks * shape_size * 4,
                ))
            contrib = sv_n_state * jv % p * xy_pow_mod[delta] % p
            out_mod[O_out_bytes] = (out_mod.get(O_out_bytes, 0) + contrib) % p

    # Decode bytes keys back to canonical-tuple form for downstream
    # consumers (`per_cell_orbit_size`, `per_cell_orbit_rep` expect
    # `Tuple[Tuple[int, ...], ...]`). One decode per UNIQUE O_out, which
    # is far fewer than per-pair (the whole point of bytes accumulation).
    shape_size = n_cells + 1
    decoded: Dict[Tuple, int] = {}
    for O_out_bytes, val in out_mod.items():
        if not O_out_bytes:
            O_out = ()
        else:
            # Unpack int sequence then group by shape_size.
            n_ints = len(O_out_bytes) // 4
            ints = list(_ffi.unpack(_ffi.cast("int*", _ffi.from_buffer(O_out_bytes)), n_ints))
            n_blocks = n_ints // shape_size
            O_out = tuple(
                tuple(ints[i * shape_size:(i + 1) * shape_size])
                for i in range(n_blocks)
            )
        decoded[O_out] = val
    return decoded


def join_partitions_c_wrapper(
    P1: Tuple[Tuple[int, ...], ...],
    P2: Tuple[Tuple[int, ...], ...],
    universe: List[int],
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Drop-in C-extension replacement for `join_partitions`.

    Returns None if C extension unavailable or universe too large
    (caller falls back to pure Python).
    """
    n_univ = len(universe)
    if n_univ == 0:
        return tuple()
    if n_univ > 256:
        return None  # exceeds C buffer; fall back

    try:
        lib, _ffi = _get_lib()
    except Exception:
        return None

    pos_to_idx = {p: i for i, p in enumerate(universe)}
    idx_to_pos = list(universe)

    p1_data = _encode_partition(P1, pos_to_idx)
    p2_data = _encode_partition(P2, pos_to_idx)

    p1_arr = _ffi.new("int[]", p1_data)
    p2_arr = _ffi.new("int[]", p2_data)

    # Output capacity: worst case = each vertex in own block = 1 + 2*n.
    # But with sizes and counts, allow generous slack.
    out_capacity = 2 * n_univ + 16
    out_arr = _ffi.new("int[]", out_capacity)

    n_out = lib.join_partitions_c(
        p1_arr, len(p1_data),
        p2_arr, len(p2_data),
        n_univ,
        out_arr, out_capacity,
    )
    if n_out < 0:
        return None  # overflow — fall back

    return _decode_partition(out_arr, n_out, idx_to_pos)
