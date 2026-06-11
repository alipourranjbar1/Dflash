/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Legacy constant for tile scheduler metadata size
// [begin_idx (inclusive), begin_block_idx (inclusive), end_idx (inclusive), end_block_idx (exclusive), begin_n_split_idx, _, _, _]
static constexpr int TileSchedulerMetaDataSize = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Legacy decode params (kept local to the FP8 extension for isolation)
struct DenseAttnDecodeParams {
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // The number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int topk;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;
    int *__restrict__ indices_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;
    index_t indices_batch_stride;
    index_t indices_row_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;
    int total_num_splits;
    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;
};

// FP8-specific extension of the legacy DenseAttnDecodeParams
struct DecodingParams_fp8 : public DenseAttnDecodeParams {
    int h_h_k_ratio;
    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(DecodingParams_fp8 &params, cudaStream_t stream);

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};
void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
