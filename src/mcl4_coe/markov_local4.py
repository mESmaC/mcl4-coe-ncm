"""Bounded Markov Chain Local-4 (MCL4) attention router and sparse QKV readout.

Unifies mcattn/fast_markov.py and oracle/markov_local4.py:
- 4 mandatory local anchor tokens {q, q-1, q-2, q-3} prevent Layer-0 byte starvation.
- Up to 12 bounded Markov-walked positions (16 total attended tokens per query)
  execute 3 relational graph hops per layer while eliminating softmax dilution.
- Bounded beam propagation (`_coalesce`) merges duplicate visited destinations
  differentiably, and a straight-through visitation-mass surrogate trains the
  four routing projections without altering forward QKV values.
"""

from __future__ import annotations

import math

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_MCL4_CUDA_EXT: Any = None
_MCL4_CUDA_EXT_ATTEMPTED: bool = False

_MCL4_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coalesce_topk_warp_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    int64_t* __restrict__ out_ids,
    float* __restrict__ out_mass,
    int32_t* __restrict__ in_to_k,
    float* __restrict__ out_denom,
    int R,
    int C,
    int N,
    int K,
    bool record_grad
) {
    extern __shared__ float smem[];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    float* row_smem = smem + warp_id * N;
    for (int i = lane; i < N; i += 32) {
        row_smem[i] = 0.0f;
    }
    __syncwarp();

    if (row >= R) return;

    const int64_t* r_ids = ids + (size_t)row * C;
    const float* r_mass = mass + (size_t)row * C;

    int local_id[6];
    float local_m[6];
    int n_items = 0;

    for (int i = lane; i < C; i += 32) {
        int id = (int)r_ids[i];
        float m = r_mass[i];
        if (id >= 0 && id < N && m > 0.0f) {
            atomicAdd(&row_smem[id], m);
        }
    }
    __syncwarp();

    for (int i = lane; i < C; i += 32) {
        int id = (int)r_ids[i];
        float m = 0.0f;
        if (id >= 0 && id < N) {
            m = atomicExch(&row_smem[id], 0.0f);
        }
        local_id[n_items] = (id >= 0 && id < N) ? id : 0;
        local_m[n_items] = m;
        n_items++;
    }
    __syncwarp();

    int64_t* o_ids = out_ids + (size_t)row * K;
    float* o_mass = out_mass + (size_t)row * K;
    float mass_sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        float best_m = -1.0f;
        int best_id = 0x7fffffff;
        int best_slot = -1;
        for (int s = 0; s < n_items; ++s) {
            float m = local_m[s];
            int id = local_id[s];
            if (m > best_m || (m == best_m && id < best_id)) {
                best_m = m;
                best_id = id;
                best_slot = s;
            }
        }
        float w_m = best_m;
        int w_id = best_id;
        int w_lane = lane;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_m = __shfl_xor_sync(0xffffffff, w_m, offset);
            int o_id = __shfl_xor_sync(0xffffffff, w_id, offset);
            int o_lane = __shfl_xor_sync(0xffffffff, w_lane, offset);
            if (o_m > w_m || (o_m == w_m && o_id < w_id) || (o_m == w_m && o_id == w_id && o_lane < w_lane)) {
                w_m = o_m;
                w_id = o_id;
                w_lane = o_lane;
            }
        }
        if (lane == w_lane && best_slot >= 0) {
            local_m[best_slot] = -2.0f;
        }
        if (lane == 0) {
            float final_m = (w_m > 0.0f) ? w_m : 0.0f;
            int final_id = (w_m > 0.0f) ? w_id : 0;
            o_ids[k] = (int64_t)final_id;
            o_mass[k] = final_m;
            mass_sum += final_m;
            if (record_grad && w_m > 0.0f) {
                row_smem[final_id] = (float)(k + 1);
            }
        }
    }
    __syncwarp();

    float denom = __shfl_sync(0xffffffff, mass_sum, 0);
    if (denom < 1e-38f) denom = 1e-38f;
    if (lane < K) {
        o_mass[lane] /= denom;
    }
    if (record_grad) {
        if (lane == 0) {
            out_denom[row] = denom;
        }
        int32_t* r_map = in_to_k + (size_t)row * C;
        for (int i = lane; i < C; i += 32) {
            int id = (int)r_ids[i];
            int k_plus_1 = (id >= 0 && id < N) ? (int)row_smem[id] : 0;
            r_map[i] = k_plus_1 - 1;
        }
        __syncwarp();
        if (lane < K) {
            int fid = (int)o_ids[lane];
            if (fid >= 0 && fid < N) row_smem[fid] = 0.0f;
        }
    }
}

__global__ void coalesce_topk_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ out_mass,
    const int32_t* __restrict__ in_to_k,
    const float* __restrict__ out_denom,
    float* __restrict__ grad_in,
    int R,
    int C,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= R) return;

    const float* g_out = grad_out + (size_t)warp_id * K;
    const float* o_m = out_mass + (size_t)warp_id * K;
    const int32_t* r_map = in_to_k + (size_t)warp_id * C;
    float* g_in = grad_in + (size_t)warp_id * C;
    float denom = out_denom[warp_id];

    float dot = 0.0f;
    if (lane < K) {
        dot = g_out[lane] * o_m[lane];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xffffffff, dot, offset);
    }
    float g_k[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        g_k[k] = (k < K) ? ((g_out[k] - dot) / denom) : 0.0f;
    }
    for (int i = lane; i < C; i += 32) {
        int k = r_map[i];
        g_in[i] = (k >= 0 && k < K) ? g_k[k] : 0.0f;
    }
}

__global__ void coalesce_sorted_small_warp_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    int64_t* __restrict__ merged_ids,
    float* __restrict__ merged_mass,
    int64_t* __restrict__ orig_groups,
    int R,
    int C
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= R) return;

    const int64_t* r_ids = ids + (size_t)warp_id * C;
    const float* r_mass = mass + (size_t)warp_id * C;
    int64_t* m_ids = merged_ids + (size_t)warp_id * C;
    float* m_mass = merged_mass + (size_t)warp_id * C;
    int64_t* o_grp = orig_groups + (size_t)warp_id * C;

    int my_id = (lane < C) ? (int)r_ids[lane] : 0x7fffffff;
    float my_m = (lane < C) ? r_mass[lane] : 0.0f;
    if (lane < C) {
        m_ids[lane] = 0;
        m_mass[lane] = 0.0f;
    }

    unsigned int active_mask = (C >= 32) ? 0xffffffffu : ((1u << C) - 1u);
    int same_mask = 0;
    for (int src = 0; src < C; ++src) {
        int other_id = __shfl_sync(0xffffffff, my_id, src);
        if (other_id == my_id) {
            same_mask |= (1 << src);
        }
    }
    int first_lane = __ffs(same_mask) - 1;
    bool is_rep = (lane < C) && (lane == first_lane);

    unsigned int rep_mask = __ballot_sync(0xffffffff, is_rep) & active_mask;
    int my_group = 0;
    for (int src = 0; src < C; ++src) {
        int other_id = __shfl_sync(0xffffffff, my_id, src);
        if ((rep_mask & (1u << src)) && (other_id < my_id)) {
            my_group++;
        }
    }
    float sum_m = 0.0f;
    for (int src = 0; src < C; ++src) {
        float other_m = __shfl_sync(0xffffffff, my_m, src);
        if (same_mask & (1 << src)) {
            sum_m += other_m;
        }
    }
    if (lane < C) {
        o_grp[lane] = (int64_t)my_group;
        if (is_rep) {
            m_ids[my_group] = (int64_t)my_id;
            m_mass[my_group] = sum_m;
        }
    }
}

__global__ void select_subset_small_warp_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    const bool* __restrict__ supported,
    const int64_t* __restrict__ positions,
    bool* __restrict__ out_mask,
    int R,
    int Q,
    int C,
    int local_tokens,
    int attend_tokens
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= R) return;

    int q_idx = warp_id % Q;
    int q_pos = (int)positions[q_idx];

    const int64_t* r_ids = ids + (size_t)warp_id * C;
    const float* r_mass = mass + (size_t)warp_id * C;
    const bool* r_sup = supported + (size_t)warp_id * C;
    bool* o_mask = out_mask + (size_t)warp_id * C;

    int my_id = (lane < C) ? (int)r_ids[lane] : 0;
    float my_m = (lane < C) ? r_mass[lane] : 0.0f;
    bool my_sup = (lane < C) ? r_sup[lane] : false;

    unsigned int active_mask = (C >= 32) ? 0xffffffffu : ((1u << C) - 1u);
    unsigned int zero_mask = __ballot_sync(0xffffffff, (lane < C) && (my_id == 0)) & active_mask;
    int first_zero = __ffs(zero_mask) - 1;
    bool is_unique = (lane < C) && ((my_id != 0) || (lane == first_zero));

    bool is_anchor = is_unique && my_sup && (my_id <= q_pos) && (my_id > q_pos - local_tokens);
    bool is_routed = (lane < C) && (my_m > 0.0f) && my_sup && (!is_anchor);

    int num_anchors = __popc(__ballot_sync(0xffffffff, is_anchor) & active_mask);
    int available = attend_tokens - num_anchors;

    int my_rank = 0;
    for (int src = 0; src < C; ++src) {
        float other_m = __shfl_sync(0xffffffff, my_m, src);
        bool other_routed = __shfl_sync(0xffffffff, (int)is_routed, src) != 0;
        if (other_routed) {
            if (other_m > my_m || (other_m == my_m && src < lane)) {
                my_rank++;
            }
        }
    }
    if (lane < C) {
        o_mask[lane] = is_anchor || (is_routed && (my_rank < available));
    }
}

std::vector<torch::Tensor> coalesce_topk_fwd_cuda(torch::Tensor ids, torch::Tensor mass, int n_tokens, int k, bool record_grad) {
    auto sizes = ids.sizes().vec();
    int C = sizes.back();
    int R = ids.numel() / C;
    sizes.back() = k;
    auto out_ids = torch::empty(sizes, ids.options());
    auto out_mass = torch::empty(sizes, mass.options());
    auto in_to_k = record_grad ? torch::empty(ids.sizes(), ids.options().dtype(torch::kInt32)) : torch::empty({0}, ids.options().dtype(torch::kInt32));
    auto out_denom = record_grad ? torch::empty({R}, mass.options()) : torch::empty({0}, mass.options());

    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = warps_per_block * n_tokens * sizeof(float);

    coalesce_topk_warp_kernel<<<blocks, threads, smem_bytes>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        out_ids.data_ptr<int64_t>(),
        out_mass.data_ptr<float>(),
        record_grad ? in_to_k.data_ptr<int32_t>() : nullptr,
        record_grad ? out_denom.data_ptr<float>() : nullptr,
        R, C, n_tokens, k, record_grad
    );
    return {out_ids, out_mass, in_to_k, out_denom};
}

torch::Tensor coalesce_topk_bwd_cuda(torch::Tensor grad_out, torch::Tensor out_mass, torch::Tensor in_to_k, torch::Tensor out_denom) {
    int C = in_to_k.size(-1);
    int K = out_mass.size(-1);
    int R = out_mass.numel() / K;
    auto grad_in = torch::empty(in_to_k.sizes(), out_mass.options());

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    coalesce_topk_bwd_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        out_mass.data_ptr<float>(),
        in_to_k.data_ptr<int32_t>(),
        out_denom.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        R, C, K
    );
    return grad_in;
}

std::vector<torch::Tensor> coalesce_sorted_small_cuda(torch::Tensor ids, torch::Tensor mass) {
    int C = ids.size(-1);
    int R = ids.numel() / C;
    auto merged_ids = torch::empty_like(ids);
    auto merged_mass = torch::empty_like(mass);
    auto orig_groups = torch::empty_like(ids);

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    coalesce_sorted_small_warp_kernel<<<blocks, threads>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        merged_ids.data_ptr<int64_t>(),
        merged_mass.data_ptr<float>(),
        orig_groups.data_ptr<int64_t>(),
        R, C
    );
    return {merged_ids, merged_mass, orig_groups};
}

torch::Tensor select_subset_small_cuda(torch::Tensor ids, torch::Tensor mass, torch::Tensor supported, torch::Tensor positions, int local_tokens, int attend_tokens) {
    int C = ids.size(-1);
    int Q = positions.numel();
    int R = ids.numel() / C;
    auto out_mask = torch::empty(ids.sizes(), ids.options().dtype(torch::kBool));

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    select_subset_small_warp_kernel<<<blocks, threads>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        supported.data_ptr<bool>(),
        positions.data_ptr<int64_t>(),
        out_mask.data_ptr<bool>(),
        R, Q, C, local_tokens, attend_tokens
    );
    return out_mask;
}

__global__ void coalesce_sorted_med_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    int64_t* __restrict__ merged_ids,
    float* __restrict__ merged_mass,
    int64_t* __restrict__ orig_groups,
    int R,
    int C
) {
    __shared__ int s_id[4][64];
    __shared__ float s_m[4][64];
    __shared__ int s_rep[4][64];

    int tid = threadIdx.x & 63;
    int row_local = threadIdx.x >> 6;
    int row = blockIdx.x * 4 + row_local;

    int my_id = 0x7fffffff;
    float my_m = 0.0f;
    if (row < R && tid < C) {
        my_id = (int)ids[(size_t)row * C + tid];
        my_m = mass[(size_t)row * C + tid];
        merged_ids[(size_t)row * C + tid] = 0;
        merged_mass[(size_t)row * C + tid] = 0.0f;
    }
    s_id[row_local][tid] = my_id;
    s_m[row_local][tid] = my_m;
    __syncthreads();

    bool is_rep = false;
    float sum_m = 0.0f;
    if (row < R && tid < C) {
        int first_src = -1;
        for (int src = 0; src < C; ++src) {
            if (s_id[row_local][src] == my_id) {
                if (first_src < 0) first_src = src;
                sum_m += s_m[row_local][src];
            }
        }
        is_rep = (tid == first_src);
    }
    s_rep[row_local][tid] = is_rep ? 1 : 0;
    __syncthreads();

    if (row < R && tid < C) {
        int my_group = 0;
        for (int src = 0; src < C; ++src) {
            if (s_rep[row_local][src] && s_id[row_local][src] < my_id) {
                my_group++;
            }
        }
        orig_groups[(size_t)row * C + tid] = (int64_t)my_group;
        if (is_rep) {
            merged_ids[(size_t)row * C + my_group] = (int64_t)my_id;
            merged_mass[(size_t)row * C + my_group] = sum_m;
        }
    }
}

__global__ void select_subset_med_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ mass,
    const bool* __restrict__ supported,
    const int64_t* __restrict__ positions,
    bool* __restrict__ out_mask,
    int R,
    int Q,
    int C,
    int local_tokens,
    int attend_tokens
) {
    __shared__ int s_id[4][64];
    __shared__ float s_m[4][64];
    __shared__ int s_anchor[4][64];
    __shared__ int s_routed[4][64];

    int tid = threadIdx.x & 63;
    int row_local = threadIdx.x >> 6;
    int row = blockIdx.x * 4 + row_local;

    int my_id = 0;
    float my_m = 0.0f;
    bool my_sup = false;
    int q_pos = 0;
    if (row < R) {
        q_pos = (int)positions[row % Q];
        if (tid < C) {
            my_id = (int)ids[(size_t)row * C + tid];
            my_m = mass[(size_t)row * C + tid];
            my_sup = supported[(size_t)row * C + tid];
        }
    }
    s_id[row_local][tid] = my_id;
    s_m[row_local][tid] = my_m;
    __syncthreads();

    bool is_anchor = false;
    bool is_routed = false;
    if (row < R && tid < C) {
        bool is_unique = (my_id != 0);
        if (!is_unique) {
            is_unique = true;
            for (int src = 0; src < tid; ++src) {
                if (s_id[row_local][src] == 0) {
                    is_unique = false;
                    break;
                }
            }
        }
        is_anchor = is_unique && my_sup && (my_id <= q_pos) && (my_id > q_pos - local_tokens);
        is_routed = (my_m > 0.0f) && my_sup && (!is_anchor);
    }
    s_anchor[row_local][tid] = is_anchor ? 1 : 0;
    s_routed[row_local][tid] = is_routed ? 1 : 0;
    __syncthreads();

    if (row < R && tid < C) {
        int num_anchors = 0;
        int my_rank = 0;
        for (int src = 0; src < C; ++src) {
            num_anchors += s_anchor[row_local][src];
            if (s_routed[row_local][src]) {
                float other_m = s_m[row_local][src];
                if (other_m > my_m || (other_m == my_m && src < tid)) {
                    my_rank++;
                }
            }
        }
        int available = attend_tokens - num_anchors;
        out_mask[(size_t)row * C + tid] = is_anchor || (is_routed && (my_rank < available));
    }
}

__global__ void markov_hop_coalesce_nograd_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ states,
    const float* __restrict__ keys,
    const int64_t* __restrict__ indices,
    const bool* __restrict__ valid,
    const bool* __restrict__ allowed,
    const int64_t* __restrict__ cur_ids,
    const float* __restrict__ cur_mass,
    const int64_t* __restrict__ seed_ids,
    const float* __restrict__ seed_mass,
    int64_t* __restrict__ out_ids,
    float* __restrict__ out_mass,
    int B, int H, int A_H, int Q, int N, int Deg, int W, int S, int K,
    float restart, float inv_sqrt_d
) {
    extern __shared__ float smem[];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;

    float* row_smem = smem + warp_id * N;
    for (int i = lane; i < N; i += 32) {
        row_smem[i] = 0.0f;
    }
    __syncwarp();

    int R = B * H * Q;
    if (row >= R) return;

    int bh = row / Q;
    int q_idx = row % Q;
    int b = bh / H;
    int h = bh % H;
    int ah = (A_H > 1) ? h : 0;

    const float* q_ptr = queries + (size_t)row * 16;
    const float* bh_states = states + (size_t)bh * N * 16;
    const float* bh_keys = keys + (size_t)bh * N * 16;
    const int64_t* bh_indices = indices + (size_t)bh * N * Deg;
    const bool* bh_valid = valid + (size_t)bh * N * Deg;
    const bool* row_allowed = allowed + (((size_t)(b * A_H + ah) * Q) + q_idx) * N;

    float q_val = (lane < 16) ? q_ptr[lane] : 0.0f;

    int cand_id[12];
    int n_cands = 0;

    const int64_t* r_cids = cur_ids + (size_t)row * W;
    const float* r_cmass = cur_mass + (size_t)row * W;

    for (int w = 0; w < W; ++w) {
        int src = (int)r_cids[w];
        float cm = r_cmass[w];
        bool src_ok = (src >= 0 && src < N) && row_allowed[src];
        if (!src_ok || cm <= 0.0f) continue;

        float s_val = (lane < 16) ? bh_states[src * 16 + lane] : 0.0f;
        float cond_d = (lane < 16) ? tanhf(q_val + s_val) : 0.0f;

        int dest = -1;
        bool perm = false;
        float score = -1e38f;
        if (lane < Deg) {
            dest = (int)bh_indices[src * Deg + lane];
            perm = bh_valid[src * Deg + lane] && (dest >= 0 && dest < N) && row_allowed[dest];
        }
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < 16; ++d) {
            float c_d = __shfl_sync(0xffffffff, cond_d, d);
            float k_d = (lane < Deg && perm) ? bh_keys[dest * 16 + d] : 0.0f;
            dot += c_d * k_d;
        }
        if (lane < Deg && perm) {
            score = dot * inv_sqrt_d;
        }
        float max_s = score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_s = __shfl_xor_sync(0xffffffff, max_s, offset);
            if (o_s > max_s) max_s = o_s;
        }
        float exp_s = (lane < Deg && perm && max_s > -1e30f) ? expf(score - max_s) : 0.0f;
        float sum_exp = exp_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
        }
        if (lane < Deg && perm && sum_exp > 0.0f) {
            float prob = exp_s / sum_exp;
            float em = (1.0f - restart) * cm * prob;
            if (em > 0.0f) {
                float prev = atomicAdd(&row_smem[dest], em);
                if (prev == 0.0f && n_cands < 12) {
                    cand_id[n_cands++] = dest;
                }
            }
        }
    }
    __syncwarp();

    const int64_t* r_sids = seed_ids + (size_t)row * S;
    const float* r_smass = seed_mass + (size_t)row * S;
    if (lane < S) {
        int sid = (int)r_sids[lane];
        float sm = restart * r_smass[lane];
        if (sid >= 0 && sid < N && sm > 0.0f) {
            float prev = atomicAdd(&row_smem[sid], sm);
            if (prev == 0.0f && n_cands < 12) {
                cand_id[n_cands++] = sid;
            }
        }
    }
    __syncwarp();

    float local_m[12];
    for (int s = 0; s < n_cands; ++s) {
        int id = cand_id[s];
        local_m[s] = atomicExch(&row_smem[id], 0.0f);
    }
    __syncwarp();

    int64_t* o_ids = out_ids + (size_t)row * K;
    float* o_mass = out_mass + (size_t)row * K;
    float mass_sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        float best_m = -1.0f;
        int best_id = 0x7fffffff;
        int best_slot = -1;
        for (int s = 0; s < n_cands; ++s) {
            float m = local_m[s];
            int id = cand_id[s];
            if (m > best_m || (m == best_m && id < best_id)) {
                best_m = m;
                best_id = id;
                best_slot = s;
            }
        }
        float w_m = best_m;
        int w_id = best_id;
        int w_lane = lane;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_m = __shfl_xor_sync(0xffffffff, w_m, offset);
            int o_id = __shfl_xor_sync(0xffffffff, w_id, offset);
            int o_lane = __shfl_xor_sync(0xffffffff, w_lane, offset);
            if (o_m > w_m || (o_m == w_m && o_id < w_id) || (o_m == w_m && o_id == w_id && o_lane < w_lane)) {
                w_m = o_m;
                w_id = o_id;
                w_lane = o_lane;
            }
        }
        if (lane == w_lane && best_slot >= 0) {
            local_m[best_slot] = -2.0f;
        }
        if (lane == 0) {
            float final_m = (w_m > 0.0f) ? w_m : 0.0f;
            int final_id = (w_m > 0.0f) ? w_id : 0;
            o_ids[k] = (int64_t)final_id;
            o_mass[k] = final_m;
            mass_sum += final_m;
        }
    }
    __syncwarp();
    float denom = __shfl_sync(0xffffffff, mass_sum, 0);
    if (denom < 1e-38f) denom = 1e-38f;
    if (lane < K) {
        o_mass[lane] /= denom;
    }
}

__global__ void markov_transitions_fwd_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ states,
    const float* __restrict__ keys,
    const int64_t* __restrict__ indices,
    const bool* __restrict__ valid,
    const bool* __restrict__ allowed,
    const int64_t* __restrict__ src_ids,
    int64_t* __restrict__ out_dest,
    float* __restrict__ out_prob,
    int B, int H, int A_H, int Q, int N, int Deg, int W,
    float inv_sqrt_d
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int R = B * H * Q;
    if (warp_id >= R) return;

    int bh = warp_id / Q;
    int q_idx = warp_id % Q;
    int b = bh / H;
    int h = bh % H;
    int ah = (A_H > 1) ? h : 0;

    const float* q_ptr = queries + (size_t)warp_id * 16;
    const float* bh_states = states + (size_t)bh * N * 16;
    const float* bh_keys = keys + (size_t)bh * N * 16;
    const int64_t* bh_indices = indices + (size_t)bh * N * Deg;
    const bool* bh_valid = valid + (size_t)bh * N * Deg;
    const bool* row_allowed = allowed + (((size_t)(b * A_H + ah) * Q) + q_idx) * N;
    const int64_t* r_sids = src_ids + (size_t)warp_id * W;

    int64_t* r_dest = out_dest + (size_t)warp_id * W * Deg;
    float* r_prob = out_prob + (size_t)warp_id * W * Deg;

    float q_val = (lane < 16) ? q_ptr[lane] : 0.0f;

    for (int w = 0; w < W; ++w) {
        int src = (int)r_sids[w];
        bool src_in = (src >= 0 && src < N);
        bool src_ok = src_in && row_allowed[src];

        int dest = 0;
        bool perm = false;
        if (lane < Deg && src_in) {
            dest = (int)bh_indices[src * Deg + lane];
            perm = src_ok && bh_valid[src * Deg + lane] && (dest >= 0 && dest < N) && row_allowed[dest];
        }
        if (lane < Deg) {
            r_dest[w * Deg + lane] = (int64_t)dest;
        }
        if (!src_ok) {
            if (lane < Deg) r_prob[w * Deg + lane] = 0.0f;
            continue;
        }

        float s_val = (lane < 16) ? bh_states[src * 16 + lane] : 0.0f;
        float cond_d = (lane < 16) ? tanhf(q_val + s_val) : 0.0f;

        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < 16; ++d) {
            float c_d = __shfl_sync(0xffffffff, cond_d, d);
            float k_d = (lane < Deg && perm) ? bh_keys[dest * 16 + d] : 0.0f;
            dot += c_d * k_d;
        }
        float score = (lane < Deg && perm) ? (dot * inv_sqrt_d) : -1e38f;
        float max_s = score;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_s = __shfl_xor_sync(0xffffffff, max_s, offset);
            if (o_s > max_s) max_s = o_s;
        }
        float exp_s = (lane < Deg && perm && max_s > -1e30f) ? expf(score - max_s) : 0.0f;
        float sum_exp = exp_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
        }
        if (lane < Deg) {
            r_prob[w * Deg + lane] = (perm && sum_exp > 0.0f) ? (exp_s / sum_exp) : 0.0f;
        }
    }
}

__global__ void markov_transitions_bwd_kernel(
    const float* __restrict__ grad_prob,
    const float* __restrict__ prob,
    const int64_t* __restrict__ dest_ids,
    const float* __restrict__ queries,
    const float* __restrict__ states,
    const float* __restrict__ keys,
    const int64_t* __restrict__ src_ids,
    float* __restrict__ grad_queries,
    float* __restrict__ grad_states,
    float* __restrict__ grad_keys,
    int B, int H, int Q, int N, int Deg, int W,
    float inv_sqrt_d
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int R = B * H * Q;
    if (warp_id >= R) return;

    int bh = warp_id / Q;

    const float* q_ptr = queries + (size_t)warp_id * 16;
    const float* bh_states = states + (size_t)bh * N * 16;
    const float* bh_keys = keys + (size_t)bh * N * 16;
    const int64_t* r_sids = src_ids + (size_t)warp_id * W;
    const int64_t* r_dest = dest_ids + (size_t)warp_id * W * Deg;
    const float* r_prob = prob + (size_t)warp_id * W * Deg;
    const float* r_gprob = grad_prob + (size_t)warp_id * W * Deg;

    float* gq_ptr = grad_queries + (size_t)warp_id * 16;
    float* bh_gstates = grad_states + (size_t)bh * N * 16;
    float* bh_gkeys = grad_keys + (size_t)bh * N * 16;

    float q_val = (lane < 16) ? q_ptr[lane] : 0.0f;
    float gq_acc = 0.0f;

    for (int w = 0; w < W; ++w) {
        int src = (int)r_sids[w];
        if (src < 0 || src >= N) continue;

        float p_j = (lane < Deg) ? r_prob[w * Deg + lane] : 0.0f;
        float g_j = (lane < Deg) ? r_gprob[w * Deg + lane] : 0.0f;
        int dest = (lane < Deg) ? (int)r_dest[w * Deg + lane] : 0;

        float dot_gp = p_j * g_j;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_gp += __shfl_xor_sync(0xffffffff, dot_gp, offset);
        }
        float ds_j = (lane < Deg && p_j > 0.0f) ? (inv_sqrt_d * p_j * (g_j - dot_gp)) : 0.0f;

        if (__ballot_sync(0xffffffff, ds_j != 0.0f) == 0u) continue;

        float s_val = (lane < 16) ? bh_states[src * 16 + lane] : 0.0f;
        float cond_d = (lane < 16) ? tanhf(q_val + s_val) : 0.0f;

        float dc_acc = 0.0f;
        for (int j = 0; j < Deg; ++j) {
            float ds = __shfl_sync(0xffffffff, ds_j, j);
            if (ds == 0.0f) continue;
            int dj = __shfl_sync(0xffffffff, dest, j);
            if (lane < 16) {
                dc_acc += ds * bh_keys[dj * 16 + lane];
                atomicAdd(&bh_gkeys[dj * 16 + lane], ds * cond_d);
            }
        }
        if (lane < 16) {
            float du = dc_acc * (1.0f - cond_d * cond_d);
            gq_acc += du;
            atomicAdd(&bh_gstates[src * 16 + lane], du);
        }
    }
    if (lane < 16) {
        gq_ptr[lane] = gq_acc;
    }
}

std::vector<torch::Tensor> coalesce_sorted_med_cuda(torch::Tensor ids, torch::Tensor mass) {
    int C = ids.size(-1);
    int R = ids.numel() / C;
    auto merged_ids = torch::empty_like(ids);
    auto merged_mass = torch::empty_like(mass);
    auto orig_groups = torch::empty_like(ids);
    int blocks = (R + 3) / 4;
    coalesce_sorted_med_kernel<<<blocks, 256>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        merged_ids.data_ptr<int64_t>(),
        merged_mass.data_ptr<float>(),
        orig_groups.data_ptr<int64_t>(),
        R, C
    );
    return {merged_ids, merged_mass, orig_groups};
}

torch::Tensor select_subset_med_cuda(torch::Tensor ids, torch::Tensor mass, torch::Tensor supported, torch::Tensor positions, int local_tokens, int attend_tokens) {
    int C = ids.size(-1);
    int Q = positions.numel();
    int R = ids.numel() / C;
    auto out_mask = torch::empty(ids.sizes(), ids.options().dtype(torch::kBool));
    int blocks = (R + 3) / 4;
    select_subset_med_kernel<<<blocks, 256>>>(
        ids.data_ptr<int64_t>(),
        mass.data_ptr<float>(),
        supported.data_ptr<bool>(),
        positions.data_ptr<int64_t>(),
        out_mask.data_ptr<bool>(),
        R, Q, C, local_tokens, attend_tokens
    );
    return out_mask;
}

std::vector<torch::Tensor> markov_hop_coalesce_nograd_cuda(
    torch::Tensor queries, torch::Tensor states, torch::Tensor keys,
    torch::Tensor indices, torch::Tensor valid, torch::Tensor allowed,
    torch::Tensor cur_ids, torch::Tensor cur_mass,
    torch::Tensor seed_ids, torch::Tensor seed_mass,
    int k, double restart, double inv_sqrt_d
) {
    int B = queries.size(0);
    int H = queries.size(1);
    int A_H = allowed.size(1);
    int Q = queries.size(2);
    int N = states.size(2);
    int Deg = indices.size(3);
    int W = cur_ids.size(3);
    int S = seed_ids.size(3);
    int R = B * H * Q;

    auto out_ids = torch::empty({B, H, Q, k}, cur_ids.options());
    auto out_mass = torch::empty({B, H, Q, k}, cur_mass.options());

    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;
    size_t smem_bytes = warps_per_block * N * sizeof(float);

    markov_hop_coalesce_nograd_kernel<<<blocks, threads, smem_bytes>>>(
        queries.data_ptr<float>(),
        states.data_ptr<float>(),
        keys.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        allowed.data_ptr<bool>(),
        cur_ids.data_ptr<int64_t>(),
        cur_mass.data_ptr<float>(),
        seed_ids.data_ptr<int64_t>(),
        seed_mass.data_ptr<float>(),
        out_ids.data_ptr<int64_t>(),
        out_mass.data_ptr<float>(),
        B, H, A_H, Q, N, Deg, W, S, k,
        (float)restart, (float)inv_sqrt_d
    );
    return {out_ids, out_mass};
}

std::vector<torch::Tensor> markov_transitions_fwd_cuda(
    torch::Tensor queries, torch::Tensor states, torch::Tensor keys,
    torch::Tensor indices, torch::Tensor valid, torch::Tensor allowed,
    torch::Tensor src_ids, double inv_sqrt_d
) {
    int B = queries.size(0);
    int H = queries.size(1);
    int A_H = allowed.size(1);
    int Q = queries.size(2);
    int N = states.size(2);
    int Deg = indices.size(3);
    int W = src_ids.size(3);
    int R = B * H * Q;

    auto out_dest = torch::empty({B, H, Q, W, Deg}, src_ids.options());
    auto out_prob = torch::empty({B, H, Q, W, Deg}, queries.options());

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    markov_transitions_fwd_kernel<<<blocks, threads>>>(
        queries.data_ptr<float>(),
        states.data_ptr<float>(),
        keys.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        allowed.data_ptr<bool>(),
        src_ids.data_ptr<int64_t>(),
        out_dest.data_ptr<int64_t>(),
        out_prob.data_ptr<float>(),
        B, H, A_H, Q, N, Deg, W, (float)inv_sqrt_d
    );
    return {out_dest, out_prob};
}

std::vector<torch::Tensor> markov_transitions_bwd_cuda(
    torch::Tensor grad_prob, torch::Tensor prob, torch::Tensor dest_ids,
    torch::Tensor queries, torch::Tensor states, torch::Tensor keys,
    torch::Tensor src_ids, double inv_sqrt_d
) {
    int B = queries.size(0);
    int H = queries.size(1);
    int Q = queries.size(2);
    int N = states.size(2);
    int Deg = prob.size(4);
    int W = src_ids.size(3);
    int R = B * H * Q;

    auto grad_queries = torch::empty_like(queries);
    auto grad_states = torch::zeros_like(states);
    auto grad_keys = torch::zeros_like(keys);

    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (R + warps_per_block - 1) / warps_per_block;

    markov_transitions_bwd_kernel<<<blocks, threads>>>(
        grad_prob.data_ptr<float>(),
        prob.data_ptr<float>(),
        dest_ids.data_ptr<int64_t>(),
        queries.data_ptr<float>(),
        states.data_ptr<float>(),
        keys.data_ptr<float>(),
        src_ids.data_ptr<int64_t>(),
        grad_queries.data_ptr<float>(),
        grad_states.data_ptr<float>(),
        grad_keys.data_ptr<float>(),
        B, H, Q, N, Deg, W, (float)inv_sqrt_d
    );
    return {grad_queries, grad_states, grad_keys};
}
"""

_MCL4_CPP_SRC = """
std::vector<torch::Tensor> coalesce_topk_fwd_cuda(torch::Tensor ids, torch::Tensor mass, int n_tokens, int k, bool record_grad);
torch::Tensor coalesce_topk_bwd_cuda(torch::Tensor grad_out, torch::Tensor out_mass, torch::Tensor in_to_k, torch::Tensor out_denom);
std::vector<torch::Tensor> coalesce_sorted_small_cuda(torch::Tensor ids, torch::Tensor mass);
torch::Tensor select_subset_small_cuda(torch::Tensor ids, torch::Tensor mass, torch::Tensor supported, torch::Tensor positions, int local_tokens, int attend_tokens);
std::vector<torch::Tensor> coalesce_sorted_med_cuda(torch::Tensor ids, torch::Tensor mass);
torch::Tensor select_subset_med_cuda(torch::Tensor ids, torch::Tensor mass, torch::Tensor supported, torch::Tensor positions, int local_tokens, int attend_tokens);
std::vector<torch::Tensor> markov_hop_coalesce_nograd_cuda(
    torch::Tensor queries, torch::Tensor states, torch::Tensor keys,
    torch::Tensor indices, torch::Tensor valid, torch::Tensor allowed,
    torch::Tensor cur_ids, torch::Tensor cur_mass,
    torch::Tensor seed_ids, torch::Tensor seed_mass,
    int k, double restart, double inv_sqrt_d
);
std::vector<torch::Tensor> markov_transitions_fwd_cuda(
    torch::Tensor queries, torch::Tensor states, torch::Tensor keys,
    torch::Tensor indices, torch::Tensor valid, torch::Tensor allowed,
    torch::Tensor src_ids, double inv_sqrt_d
);
std::vector<torch::Tensor> markov_transitions_bwd_cuda(
    torch::Tensor grad_prob, torch::Tensor prob, torch::Tensor dest_ids,
    torch::Tensor queries, torch::Tensor states, torch::Tensor keys,
    torch::Tensor src_ids, double inv_sqrt_d
);
"""


def _get_mcl4_cuda_ext() -> Any:
    """Lazy-compile or load cached CUDA warp extension for MarkovLocal4Router."""
    global _MCL4_CUDA_EXT, _MCL4_CUDA_EXT_ATTEMPTED
    if _MCL4_CUDA_EXT_ATTEMPTED:
        return _MCL4_CUDA_EXT
    _MCL4_CUDA_EXT_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None
    try:
        import shutil
        import subprocess
        import sys

        if sys.platform == "win32" and shutil.which("cl") is None:
            return None
        if shutil.which("ninja") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=False)
        from torch.utils.cpp_extension import load_inline

        _MCL4_CUDA_EXT = load_inline(
            name="mcl4_fast_ext_v4",
            cpp_sources=_MCL4_CPP_SRC,
            cuda_sources=_MCL4_CUDA_SRC,
            functions=[
                "coalesce_topk_fwd_cuda",
                "coalesce_topk_bwd_cuda",
                "coalesce_sorted_small_cuda",
                "select_subset_small_cuda",
                "coalesce_sorted_med_cuda",
                "select_subset_med_cuda",
                "markov_hop_coalesce_nograd_cuda",
                "markov_transitions_fwd_cuda",
                "markov_transitions_bwd_cuda",
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _MCL4_CUDA_EXT = None
    return _MCL4_CUDA_EXT


class _CoalesceTopKAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, ids: Tensor, mass: Tensor, n_tokens: int, k: int) -> tuple[Tensor, Tensor]:
        ext = _get_mcl4_cuda_ext()
        req = bool(mass.requires_grad)
        out_ids, out_mass, in_to_k, out_denom = ext.coalesce_topk_fwd_cuda(
            ids.contiguous(), mass.float().contiguous(), int(n_tokens), int(k), req
        )
        if req:
            ctx.save_for_backward(out_mass, in_to_k, out_denom)
        return out_ids, out_mass.to(mass.dtype)

    @staticmethod
    def backward(ctx: Any, grad_ids: Tensor | None, grad_mass: Tensor) -> tuple[None, Tensor, None, None]:
        ext = _get_mcl4_cuda_ext()
        out_mass, in_to_k, out_denom = ctx.saved_tensors
        grad_in = ext.coalesce_topk_bwd_cuda(
            grad_mass.float().contiguous(), out_mass, in_to_k, out_denom
        )
        return None, grad_in.to(grad_mass.dtype), None, None


class _CoalesceSmallAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, ids: Tensor, mass: Tensor) -> tuple[Tensor, Tensor]:
        ext = _get_mcl4_cuda_ext()
        if ids.shape[-1] <= 32:
            m_ids, m_mass, o_grp = ext.coalesce_sorted_small_cuda(
                ids.contiguous(), mass.float().contiguous()
            )
        else:
            m_ids, m_mass, o_grp = ext.coalesce_sorted_med_cuda(
                ids.contiguous(), mass.float().contiguous()
            )
        if mass.requires_grad:
            ctx.save_for_backward(o_grp)
        return m_ids, m_mass.to(mass.dtype)

    @staticmethod
    def backward(ctx: Any, grad_ids: Tensor | None, grad_mass: Tensor) -> tuple[None, Tensor]:
        (o_grp,) = ctx.saved_tensors
        return None, grad_mass.gather(-1, o_grp)


class _MarkovTransitionsAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        queries: Tensor,
        states: Tensor,
        keys: Tensor,
        indices: Tensor,
        valid: Tensor,
        allowed: Tensor,
        src_ids: Tensor,
        inv_sqrt_d: float,
    ) -> tuple[Tensor, Tensor]:
        ext = _get_mcl4_cuda_ext()
        q_c = queries.float().contiguous()
        s_c = states.float().contiguous()
        k_c = keys.float().contiguous()
        idx_c = indices.contiguous()
        val_c = valid.contiguous()
        all_c = allowed.contiguous()
        src_c = src_ids.contiguous()
        dest, prob = ext.markov_transitions_fwd_cuda(
            q_c, s_c, k_c, idx_c, val_c, all_c, src_c, float(inv_sqrt_d)
        )
        if queries.requires_grad or states.requires_grad or keys.requires_grad:
            ctx.save_for_backward(prob, dest, q_c, s_c, k_c, src_c)
            ctx.inv_sqrt_d = float(inv_sqrt_d)
        return dest, prob.to(queries.dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_dest: Tensor | None, grad_prob: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None, None, None]:
        ext = _get_mcl4_cuda_ext()
        prob, dest, q_c, s_c, k_c, src_c = ctx.saved_tensors
        gq, gs, gk = ext.markov_transitions_bwd_cuda(
            grad_prob.float().contiguous(), prob, dest, q_c, s_c, k_c, src_c, ctx.inv_sqrt_d
        )
        return (
            gq.to(grad_prob.dtype),
            gs.to(grad_prob.dtype),
            gk.to(grad_prob.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def coalesce_candidates(ids: Tensor, mass: Tensor) -> tuple[Tensor, Tensor]:
    """Sum duplicate token visitation masses and return sorted unique IDs with zero padding."""
    if ids.is_cuda and ids.shape[-1] <= 64 and _get_mcl4_cuda_ext() is not None:
        return _CoalesceSmallAutograd.apply(ids, mass)
    if ids.is_cuda:
        sorted_ids_i32, order = ids.int().sort(dim=-1, stable=False)
        sorted_ids = sorted_ids_i32.long()
    else:
        sorted_ids, order = ids.sort(dim=-1, stable=True)
    sorted_mass = mass.gather(-1, order)
    boundary = torch.ones_like(sorted_ids, dtype=torch.bool)
    boundary[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    groups = boundary.long().cumsum(-1) - 1
    merged_ids = torch.zeros_like(ids).scatter(-1, groups, sorted_ids)
    merged_mass = torch.zeros_like(mass).scatter_add(-1, groups, sorted_mass)
    return merged_ids, merged_mass


def _gather_nodes(nodes: Tensor, ids: Tensor) -> Tensor:
    """Gather [B, H, N, D] node features at [B, H, ...] indices via embedding lookup."""
    b, h, n, d = nodes.shape
    offsets = torch.arange(b * h, device=ids.device).reshape(b, h, *([1] * (ids.ndim - 2))) * n
    return F.embedding(ids + offsets, nodes.reshape(b * h * n, d))


def _supported(allowed: Tensor, ids: Tensor) -> Tensor:
    """Gather query-specific support [B, 1 or H, Q, N] at candidate IDs."""
    if allowed.shape[1] == 1 and ids.shape[1] > 1:
        allowed = allowed.expand(-1, ids.shape[1], -1, -1)
    return allowed.gather(-1, ids.flatten(3)).reshape_as(ids)


class MarkovLocal4Router(nn.Module):
    """Bounded Markov graph walker with 4 mandatory local anchors."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        router_dim: int = 16,
        graph_degree: int = 4,
        num_walkers: int = 4,
        steps: int = 3,
        attend_tokens: int = 16,
        local_tokens: int = 4,
        beam_size: int = 8,
        restart: float = 0.1,
        query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if local_tokens < 0 or local_tokens > attend_tokens:
            raise ValueError("local_tokens must be between 0 and attend_tokens")
        self.d_model = d_model
        self.n_heads = n_heads
        self.router_dim = router_dim
        self.graph_degree = graph_degree
        self.num_walkers = num_walkers
        self.steps = steps
        self.attend_tokens = attend_tokens
        self.local_tokens = local_tokens
        self.beam_size = beam_size
        self.restart = restart
        self.query_chunk_size = query_chunk_size

        for name in ("route_query", "route_state", "route_key", "start_query"):
            setattr(self, name, nn.Linear(d_model, n_heads * router_dim, bias=False))
        self._graph_cache: dict[
            tuple[int, torch.device, int], tuple[Tensor, Tensor, Tensor, int, Tensor]
        ] = {}

    def _project(self, x: Tensor, name: str) -> Tensor:
        b, n, _ = x.shape
        out = getattr(self, name)(x).view(b, n, self.n_heads, self.router_dim).transpose(1, 2)
        return out.float().contiguous() if out.is_cuda else out.float()

    def project_routes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.is_cuda:
            b, n, _ = x.shape
            w_cat = torch.cat(
                (
                    self.route_query.weight,
                    self.route_state.weight,
                    self.route_key.weight,
                    self.start_query.weight,
                ),
                dim=0,
            )
            out = F.linear(x, w_cat).view(b, n, 4, self.n_heads, self.router_dim).float()
            out = out.permute(2, 0, 3, 1, 4).contiguous()
            return out[0], out[1], out[2], out[3]
        return (
            self._project(x, "route_query"),
            self._project(x, "route_state"),
            self._project(x, "route_key"),
            self._project(x, "start_query"),
        )

    @torch.no_grad()
    def build_graph(self, keys: Tensor, key_valid: Tensor) -> tuple[Tensor, Tensor]:
        """Construct bounded-degree causal token graph (7 local offsets + top-k semantic earlier)."""
        b, h, n, _ = keys.shape
        with torch.autocast(device_type=keys.device.type, enabled=False):
            cache_key = (n, keys.device, self.graph_degree)
            cached = getattr(self, "_graph_cache", {}).get(cache_key)
            if cached is None:
                position = torch.arange(n, device=keys.device)
                earlier = (position[None, :] < position[:, None])[None, None]
                offsets = torch.tensor(
                    [0, -1, 1, -2, 2, -3, -4, 4, -8, -16, -32, -64], device=keys.device
                )
                raw = position[:, None] + offsets
                local_clamped = raw.clamp(0, n - 1)
                raw_in_bounds = ((raw >= 0) & (raw < n))[None, None]
                semantic_k = min(n, self.graph_degree)
                degree = 12 + semantic_k
                earlier_slot = torch.ones(
                    degree, degree, device=keys.device, dtype=torch.bool
                ).tril(-1)
                cached = (earlier, local_clamped, raw_in_bounds, semantic_k, earlier_slot)
                if hasattr(self, "_graph_cache"):
                    self._graph_cache[cache_key] = cached
            earlier, local_clamped, raw_in_bounds, semantic_k, earlier_slot = cached

            sim_dtype = torch.bfloat16 if keys.is_cuda else torch.float32
            normalized = F.normalize(keys.float(), dim=-1).to(sim_dtype)
            similarity = normalized @ normalized.transpose(-1, -2)
            eligible = earlier & key_valid[:, None, None, :] & key_valid[:, None, :, None]
            similarity = similarity.masked_fill(~eligible, -torch.inf)
            semantic = torch.topk(similarity, k=semantic_k, dim=-1).indices
            semantic_valid = eligible.expand(b, h, -1, -1).gather(-1, semantic)

            local = local_clamped[None, None].expand(b, h, -1, -1)
            node_valid_1 = key_valid[:, None, :, None]
            dest_valid = key_valid[:, local_clamped].unsqueeze(1)
            local_valid = (raw_in_bounds & dest_valid & node_valid_1).expand(b, h, -1, -1).clone()
            local_valid[..., 0] = True

            indices = torch.cat((local, semantic), dim=-1)
            valid = torch.cat((local_valid, semantic_valid), dim=-1)
            duplicate = (
                (indices.unsqueeze(-1) == indices.unsqueeze(-2))
                & earlier_slot
                & valid.unsqueeze(-2)
            ).any(-1)
            return indices, valid & ~duplicate

    def _allowed_nodes(
        self,
        key_valid: Tensor,
        positions: Tensor,
        causal: bool,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        n = key_valid.shape[-1]
        allowed = key_valid[:, None, None, :].expand(-1, 1, positions.numel(), -1)
        if causal:
            allowed = allowed & (torch.arange(n, device=positions.device) <= positions[:, None])
        if attn_mask is not None:
            allowed = allowed & attn_mask[:, :, positions, :]
        fallback = torch.arange(n, device=positions.device) == positions[:, None]
        return allowed | (~allowed.any(-1, keepdim=True) & fallback)

    def sparse_seeds(self, starts: Tensor, keys: Tensor, allowed: Tensor) -> tuple[Tensor, Tensor]:
        k = min(self.num_walkers, keys.shape[-2])
        if keys.is_cuda:
            with torch.no_grad():
                scores_bf16 = (
                    starts.to(torch.bfloat16) @ keys.to(torch.bfloat16).transpose(-1, -2)
                ).masked_fill(~allowed, float("-inf"))
                ids = torch.topk(scores_bf16, k=k, dim=-1).indices
            with torch.autocast(device_type=keys.device.type, enabled=False):
                sel_keys = _gather_nodes(keys.float(), ids)
                top_scores = torch.matmul(
                    starts.float().unsqueeze(-2), sel_keys.transpose(-1, -2)
                ).squeeze(-2) / math.sqrt(self.router_dim)
                top_scores = top_scores.masked_fill(~_supported(allowed, ids), -torch.inf)
                return ids, top_scores.softmax(-1)
        with torch.autocast(device_type=keys.device.type, enabled=False):
            scores = (starts.float() @ keys.float().transpose(-1, -2)) / math.sqrt(self.router_dim)
            scores = scores.masked_fill(~allowed, -torch.inf)
            top_scores, ids = torch.topk(scores, k=k, dim=-1)
            return ids, top_scores.softmax(-1)

    def transitions_at_sources(
        self,
        queries: Tensor,
        states: Tensor,
        indices: Tensor,
        valid: Tensor,
        allowed: Tensor,
        source_ids: Tensor,
        neighbor_key_rows: Tensor | None = None,
        *,
        keys: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        ext = _get_mcl4_cuda_ext() if queries.is_cuda else None
        if (
            ext is not None
            and keys is not None
            and self.router_dim == 16
            and indices.shape[-1] <= 32
            and source_ids.shape[-1] <= 8
        ):
            return _MarkovTransitionsAutograd.apply(
                queries,
                states,
                keys,
                indices,
                valid,
                allowed,
                source_ids,
                1.0 / math.sqrt(self.router_dim),
            )
        destinations = _gather_nodes(indices, source_ids)
        permitted = (
            _gather_nodes(valid, source_ids)
            & _supported(allowed, destinations)
            & _supported(allowed, source_ids).unsqueeze(-1)
        )
        if neighbor_key_rows is None:
            if keys is None:
                raise ValueError("Either neighbor_key_rows or keys must be provided")
            nk_dtype = torch.bfloat16 if keys.is_cuda else torch.float32
            neighbor_key_rows = _gather_nodes(keys.to(nk_dtype), indices).flatten(-2)
        if queries.is_cuda:
            q_bf = queries.to(torch.bfloat16)
            source_states = _gather_nodes(states.to(torch.bfloat16), source_ids)
            neighbor_keys = _gather_nodes(neighbor_key_rows, source_ids).reshape(
                *destinations.shape, self.router_dim
            )
            condition = torch.tanh(q_bf.unsqueeze(-2) + source_states)
            scores = (
                torch.matmul(condition.unsqueeze(-2), neighbor_keys.transpose(-1, -2))
                .squeeze(-2)
                .float()
                / math.sqrt(self.router_dim)
            )
            scores = scores.masked_fill(~permitted, -torch.inf)
            scores = torch.where(permitted.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            return destinations, scores.softmax(-1).masked_fill(~permitted, 0)
        with torch.autocast(device_type=queries.device.type, enabled=False):
            source_states = _gather_nodes(states.float(), source_ids)
            neighbor_keys = _gather_nodes(neighbor_key_rows, source_ids).reshape(
                *destinations.shape, self.router_dim
            )
            condition = torch.tanh(queries.float().unsqueeze(-2) + source_states)
            scores = torch.matmul(
                condition.unsqueeze(-2), neighbor_keys.transpose(-1, -2)
            ).squeeze(-2) / math.sqrt(self.router_dim)
            scores = scores.masked_fill(~permitted, -torch.inf)
            scores = torch.where(permitted.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            return destinations, scores.softmax(-1).masked_fill(~permitted, 0)

    def bounded_visits(
        self,
        queries: Tensor,
        states: Tensor,
        n_tokens: int,
        indices: Tensor,
        valid: Tensor,
        allowed: Tensor,
        seed_ids: Tensor,
        seed_mass: Tensor,
        neighbor_key_rows: Tensor | None = None,
        *,
        keys: Tensor | None = None,
        coalesce_final: bool = True,
    ) -> tuple[Tensor, Tensor]:
        current_ids, current_mass = seed_ids, seed_mass
        visited_ids, visited_mass = [seed_ids], [seed_mass]
        ext = _get_mcl4_cuda_ext() if queries.is_cuda else None
        inv_sqrt_d = 1.0 / math.sqrt(self.router_dim)
        for step_i in range(self.steps):
            is_last = step_i == (self.steps - 1)
            need_grad = is_last and self.training and torch.is_grad_enabled()
            width = min(self.beam_size, n_tokens)
            if (
                not need_grad
                and ext is not None
                and keys is not None
                and self.router_dim == 16
                and n_tokens <= 2048
                and width <= 16
                and indices.shape[-1] <= 32
                and current_ids.shape[-1] <= 8
                and seed_ids.shape[-1] <= 8
            ):
                with torch.no_grad():
                    current_ids, current_mass = ext.markov_hop_coalesce_nograd_cuda(
                        queries.float().contiguous(),
                        states.float().contiguous(),
                        keys.float().contiguous(),
                        indices.contiguous(),
                        valid.contiguous(),
                        allowed.contiguous(),
                        current_ids.contiguous(),
                        current_mass.float().contiguous(),
                        seed_ids.contiguous(),
                        seed_mass.float().contiguous(),
                        int(width),
                        float(self.restart),
                        float(inv_sqrt_d),
                    )
                    current_mass = current_mass.to(seed_mass.dtype)
            else:
                ctx = torch.enable_grad() if need_grad else torch.no_grad()
                with ctx:
                    c_mass_in = current_mass.detach() if is_last else current_mass
                    s_mass_in = seed_mass if is_last else seed_mass.detach()
                    destinations, probabilities = self.transitions_at_sources(
                        queries,
                        states,
                        indices,
                        valid,
                        allowed,
                        current_ids,
                        neighbor_key_rows,
                        keys=keys,
                    )
                    candidate_ids = torch.cat((destinations.flatten(-2), seed_ids), dim=-1)
                    edge_mass = (1.0 - self.restart) * c_mass_in.unsqueeze(-1) * probabilities
                    candidate_mass = torch.cat(
                        (edge_mass.flatten(-2), self.restart * s_mass_in), dim=-1
                    )
                    width = min(self.beam_size, n_tokens, candidate_ids.shape[-1])
                    if (
                        ext is not None
                        and n_tokens <= 2048
                        and width <= 16
                        and candidate_ids.shape[-1] <= 192
                    ):
                        current_ids, current_mass = _CoalesceTopKAutograd.apply(
                            candidate_ids, candidate_mass, n_tokens, width
                        )
                    else:
                        if n_tokens <= 2 * candidate_ids.shape[-1]:
                            merged_mass = candidate_mass.new_zeros(
                                *candidate_mass.shape[:-1], n_tokens
                            )
                            merged_mass = merged_mass.scatter_add(
                                -1, candidate_ids, candidate_mass
                            )
                            merged_ids = torch.arange(n_tokens, device=queries.device).expand_as(
                                merged_mass
                            )
                        else:
                            merged_ids, merged_mass = coalesce_candidates(
                                candidate_ids, candidate_mass
                            )
                        width = min(self.beam_size, n_tokens, merged_mass.shape[-1])
                        if queries.is_cuda:
                            current_mass, chosen = torch.topk(
                                merged_mass, k=width, dim=-1, largest=True, sorted=True
                            )
                            current_ids = merged_ids.gather(-1, chosen)
                        else:
                            chosen = merged_mass.argsort(dim=-1, descending=True, stable=True)[
                                ..., :width
                            ]
                            current_ids = merged_ids.gather(-1, chosen)
                            current_mass = merged_mass.gather(-1, chosen)
                        current_mass = current_mass / current_mass.sum(-1, keepdim=True).clamp_min(
                            torch.finfo(current_mass.dtype).tiny
                        )
            visited_ids.append(current_ids)
            visited_mass.append(current_mass)
        all_ids = torch.cat(visited_ids, dim=-1)
        all_mass = torch.cat(visited_mass, dim=-1) / (self.steps + 1)
        if not coalesce_final:
            return all_ids, all_mass
        return coalesce_candidates(all_ids, all_mass)

    def select_subset(
        self, ids: Tensor, mass: Tensor, positions: Tensor, allowed: Tensor
    ) -> Tensor:
        """Reserve up to `local_tokens` local anchors, then rank distinct Markov-visited tokens."""
        ext = _get_mcl4_cuda_ext() if ids.is_cuda else None
        if ext is not None and ids.shape[-1] <= 64:
            sup = _supported(allowed, ids)
            if ids.shape[-1] <= 32:
                return ext.select_subset_small_cuda(
                    ids.contiguous(),
                    mass.float().contiguous(),
                    sup.contiguous(),
                    positions.long().contiguous(),
                    int(self.local_tokens),
                    int(self.attend_tokens),
                )
            return ext.select_subset_med_cuda(
                ids.contiguous(),
                mass.float().contiguous(),
                sup.contiguous(),
                positions.long().contiguous(),
                int(self.local_tokens),
                int(self.attend_tokens),
            )
        query = positions.view(1, 1, -1, 1)
        zero = ids == 0
        unique = ~zero | (zero.long().cumsum(-1) == 1)
        supported = _supported(allowed, ids)
        anchors = unique & supported & (ids <= query) & (ids > query - self.local_tokens)
        routed = (mass > 0) & supported & ~anchors
        k_sel = min(self.attend_tokens, ids.shape[-1])
        if ids.is_cuda:
            ranked = torch.topk(
                mass.masked_fill(~routed, -torch.inf), k=k_sel, dim=-1, largest=True, sorted=True
            ).indices
        else:
            ranked = mass.masked_fill(~routed, -torch.inf).argsort(
                dim=-1, descending=True, stable=True
            )[..., :k_sel]
        available = self.attend_tokens - anchors.sum(-1, keepdim=True)
        take = torch.arange(ranked.shape[-1], device=ids.device) < available
        take = take & routed.gather(-1, ranked)
        return anchors | torch.zeros_like(anchors).scatter(-1, ranked, take)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        x_norm: Tensor,
        key_valid: Tensor,
        *,
        causal: bool = True,
        dropout_p: float = 0.0,
        attn_scale: float | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Route and compute sparse QKV attention over 4 local anchors + 12 Markov walkers."""
        b, h, n, head_dim = q.shape
        scale_val = float(attn_scale) if attn_scale is not None else (1.0 / math.sqrt(head_dim))
        queries, states, keys, starts = self.project_routes(x_norm)
        indices, valid = self.build_graph(keys, key_valid)
        ext = _get_mcl4_cuda_ext() if keys.is_cuda else None
        use_fast_cuda_hops = (
            ext is not None
            and self.router_dim == 16
            and indices.shape[-1] <= 32
            and self.beam_size <= 8
            and self.num_walkers <= 8
        )
        if use_fast_cuda_hops:
            neighbor_key_rows = None
        else:
            nk_dtype = torch.bfloat16 if keys.is_cuda else torch.float32
            neighbor_key_rows = _gather_nodes(keys.to(nk_dtype), indices).flatten(-2)
        chunks = []
        for start in range(0, n, self.query_chunk_size):
            end = min(start + self.query_chunk_size, n)
            positions = torch.arange(start, end, device=q.device)
            allowed = self._allowed_nodes(key_valid, positions, causal=causal, attn_mask=attn_mask)
            seed_ids, seed_mass = self.sparse_seeds(starts[:, :, start:end], keys, allowed)
            ids, mass = self.bounded_visits(
                queries[:, :, start:end],
                states,
                n,
                indices,
                valid,
                allowed,
                seed_ids,
                seed_mass,
                neighbor_key_rows,
                keys=keys,
                coalesce_final=(self.local_tokens <= 0),
            )
            if self.local_tokens > 0:
                anchors = (
                    positions[:, None] - torch.arange(self.local_tokens, device=q.device)
                ).clamp_min(0)
                anchors = anchors.expand(b, h, -1, -1)
                ids, mass = coalesce_candidates(
                    torch.cat((ids, anchors), dim=-1),
                    torch.cat((mass, torch.zeros_like(anchors, dtype=mass.dtype)), dim=-1),
                )
            q_chunk = q[:, :, start:end]
            use_dense = (
                q.is_cuda
                and q_chunk.shape[-2] > 1
                and n <= 2 * ids.shape[-1] * head_dim
            )
            if use_dense:
                scores = torch.matmul(q_chunk, k.transpose(-1, -2)).gather(-1, ids)
            else:
                selected_k = _gather_nodes(k, ids)
                scores = torch.matmul(q_chunk.unsqueeze(-2), selected_k.transpose(-1, -2)).squeeze(
                    -2
                )
            scores = scores.float() * scale_val
            hard_mask = self.select_subset(ids, mass, positions, allowed)
            with torch.autocast(device_type=q.device.type, enabled=False):
                hard_weights = scores.masked_fill(~hard_mask, -torch.inf).softmax(-1)
                if self.training and torch.is_grad_enabled():
                    soft_allowed = (mass > 0) & _supported(allowed, ids)
                    soft_logits = (
                        scores.detach() + mass.clamp_min(torch.finfo(torch.float32).tiny).log()
                    )
                    soft_weights = soft_logits.masked_fill(~soft_allowed, -torch.inf).softmax(-1)
                    weights = hard_weights + (soft_weights - soft_weights.detach())
                else:
                    weights = hard_weights
            if dropout_p > 0.0 and self.training:
                weights = F.dropout(weights, p=dropout_p, training=True)
            if use_dense:
                cand_w = weights.to(v.dtype)
                dense_w = cand_w.new_zeros(*weights.shape[:-1], n).scatter_add(-1, ids, cand_w)
                attended = torch.matmul(dense_w, v)
            else:
                selected_v = _gather_nodes(v, ids)
                attended = torch.matmul(weights.to(v.dtype).unsqueeze(-2), selected_v).squeeze(-2)
            chunks.append(attended)
        return torch.cat(chunks, dim=2)
