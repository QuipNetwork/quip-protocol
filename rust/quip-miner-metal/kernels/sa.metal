// SPDX-License-Identifier: AGPL-3.0-or-later
// Metal simulated annealing — explicit per-job (no nonce/slot economy).
// One thread = one independent read. Host uploads one Ising problem (CSR + h)
// and a beta schedule; kernel writes final spins for each read.
//
// Ported algorithm semantics from quip-miner-cuda/kernels/sa.cu (Metropolis,
// geometric beta ladder, xorshift32 RNG).

#include <metal_stdlib>
using namespace metal;

constant float RNG_SCALE = 2.32830643653869628906e-10f; // 1.0f / 2^32

struct KernelParams {
    int num_betas;
    int sweeps_per_beta;
    int num_reads;
    int N;
    uint base_seed;
};

// Buffer layout (must match sampler.rs set_buffer indices):
// 0 row_ptr, 1 col_ind, 2 j_vals, 3 h_vals, 4 workspace, 5 out_spins,
// 6 beta_schedule, 7 params

inline uint xorshift32(thread uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

inline float effective_field(
    int var,
    device const char *state,
    device const int *row_ptr,
    device const int *col_ind,
    device const float *j_vals,
    device const float *h_vals
) {
    float heff = h_vals[var];
    int start = row_ptr[var];
    int end = row_ptr[var + 1];
    for (int p = start; p < end; ++p) {
        int nbr = col_ind[p];
        heff += j_vals[p] * (float)state[nbr];
    }
    return heff;
}

inline float flip_delta(
    int var,
    device const char *state,
    device const int *row_ptr,
    device const int *col_ind,
    device const float *j_vals,
    device const float *h_vals
) {
    float s = (float)state[var];
    return -2.0f * s * effective_field(
        var, state, row_ptr, col_ind, j_vals, h_vals);
}

kernel void metal_sa_sample(
    device const int *row_ptr [[buffer(0)]],
    device const int *col_ind [[buffer(1)]],
    device const float *j_vals [[buffer(2)]],
    device const float *h_vals [[buffer(3)]],
    device char *workspace [[buffer(4)]],
    device char *out_spins [[buffer(5)]],
    device const float *beta_schedule [[buffer(6)]],
    constant KernelParams &params [[buffer(7)]],
    uint read [[thread_position_in_grid]]
) {
    if (read >= (uint)params.num_reads) return;

    int N = params.N;
    device char *state = workspace + (size_t)read * (size_t)N;
    device char *out = out_spins + (size_t)read * (size_t)N;

    uint rng = params.base_seed
        ^ (uint)((read + 1u) * 12345u)
        ^ 0x9E3779B9u;
    if (rng == 0u) rng = 0xdeadbeefu;

    for (int var = 0; var < N; ++var) {
        uint r = xorshift32(rng);
        state[var] = (r & 1u) ? (char)-1 : (char)1;
    }

    if (N == 0) return;

    for (int bi = 0; bi < params.num_betas; ++bi) {
        float beta = beta_schedule[bi];
        float threshold = 22.18f / max(beta, 1e-12f);

        for (int sweep = 0; sweep < params.sweeps_per_beta; ++sweep) {
            for (int var = 0; var < N; ++var) {
                float de = flip_delta(
                    var, state, row_ptr, col_ind, j_vals, h_vals);
                if (de >= threshold) {
                    continue;
                }
                bool flip = false;
                if (de <= 0.0f) {
                    flip = true;
                } else {
                    float accept = metal::precise::exp(-de * beta);
                    float u = (float)xorshift32(rng) * RNG_SCALE;
                    flip = (accept > u);
                }
                if (flip) {
                    state[var] = (char)(-state[var]);
                }
            }
        }
    }

    for (int var = 0; var < N; ++var) {
        out[var] = state[var];
    }
}
