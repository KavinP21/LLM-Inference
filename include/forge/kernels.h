#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace forge {

void launch_embedding(const std::int32_t* tokens, const half* table, half* output,
                      int token_count, int hidden_size, cudaStream_t stream);
void launch_rmsnorm(const half* input, const half* weight, half* output,
                    int rows, int hidden_size, float epsilon, cudaStream_t stream);
void launch_residual_rmsnorm(half* residual, const half* update, const half* weight,
                             half* normalized, int rows, int hidden_size,
                             float epsilon, cudaStream_t stream);
void launch_add(half* destination, const half* update, std::int64_t elements,
                cudaStream_t stream);
void launch_silu_mul(const half* gate, const half* up, half* output,
                     std::int64_t elements, cudaStream_t stream);
void launch_rope_cache_write(
    half* query, half* key, const half* value, half* kv_cache,
    const std::int32_t* positions, const std::int32_t* block_table,
    int batch, int query_heads, int kv_heads, int head_dim, int max_blocks,
    int block_tokens, int layer, int num_layers, float rope_theta, cudaStream_t stream);
void launch_rope_cache_write_prefill(
    half* query, half* key, const half* value, half* kv_cache,
    const std::int32_t* block_table, int token_count, int query_heads,
    int kv_heads, int head_dim, int block_tokens, int layer,
    int num_layers, float rope_theta, cudaStream_t stream);
void launch_paged_decode_attention(
    const half* query, const half* kv_cache, half* output,
    const std::int32_t* context_lengths, const std::int32_t* block_table,
    int batch, int query_heads, int kv_heads, int head_dim, int max_blocks,
    int block_tokens, int layer, int num_layers, cudaStream_t stream);
void launch_paged_prefill_attention(
    const half* query, const half* kv_cache, half* output,
    const std::int32_t* block_table, int token_count, int query_heads,
    int kv_heads, int head_dim, int block_tokens, int layer,
    int num_layers, cudaStream_t stream);
void launch_causal_softmax(half* scores, const std::int32_t* lengths,
                           int batch, int heads, int query_length, int key_length,
                           cudaStream_t stream);
void launch_argmax(const half* logits, std::int32_t* tokens,
                   int batch, int vocab_size, cudaStream_t stream);

}  // namespace forge
