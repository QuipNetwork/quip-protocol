// SPDX-License-Identifier: AGPL-3.0-or-later
// Double-precision Ising energy → milli-int (trunc toward zero).
// Used by golden-parity tests to evaluate fixed spin configs on the GPU.
// Convention matches quip_protocol::scoring::energy_milli:
//   E = sum_i h_i s_i + sum_edges J_uv s_u s_v
//   energy_milli = (long long)(E * 1000.0)   // truncation toward zero

extern "C" {

// One block, thread 0 computes full energy (problems are small in golden set).
// spins: N values in {-1,+1} as signed char
// h: N doubles, j: M doubles, edges_u/v: M ints
// out_energy_milli: single i64
__global__ void cuda_energy_milli(
    const signed char* __restrict__ spins,
    const double* __restrict__ h,
    const double* __restrict__ j,
    const int* __restrict__ edges_u,
    const int* __restrict__ edges_v,
    int N,
    int M,
    long long* __restrict__ out_energy_milli
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double e = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = (spins[i] > 0) ? 1.0 : -1.0;
        e += h[i] * s;
    }
    for (int k = 0; k < M; ++k) {
        int u = edges_u[k];
        int v = edges_v[k];
        if (u < 0 || v < 0 || u >= N || v >= N) {
            continue;  // OOB edge skipped (matches energy_milli)
        }
        double su = (spins[u] > 0) ? 1.0 : -1.0;
        double sv = (spins[v] > 0) ? 1.0 : -1.0;
        e += j[k] * su * sv;
    }

    if (!isfinite(e)) {
        *out_energy_milli = 1LL << 62;
        return;
    }
    // Truncation toward zero, identical to Rust `(e * 1000.0) as i64`.
    *out_energy_milli = (long long)(e * 1000.0);
}

}  // extern "C"
