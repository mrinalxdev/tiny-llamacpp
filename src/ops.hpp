#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "gguf.hpp"

namespace tinyinfer {

struct Mat {
    GgmlType type{};
    const std::byte* data = nullptr;
    std::uint64_t cols = 0;
    std::uint64_t rows = 0;
    std::uint64_t row_bytes = 0;

    static Mat from(const Gguf& g, std::string_view name);
    const std::byte* row(std::uint64_t r) const { return data + r * row_bytes; }
};


/*
Scalar reference implementations of every operation the Qwen2 forward pass needs.
Deliberately simple and slow: they are the "ground truth" that the optimized
(SIMD / threaded) versions in Step 5 will be tested against.
*/

void rmsnorm(float* out, const float* x, const float* w, std::size_t n, float eps);
void softmax(float* x, std::size_t n);
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
void silu_mul(float* gate, const float* up, std::size_t n);   
void add_inplace(float* out, const float* x, std::size_t n);
float dot(const float* a, const float* b, std::size_t n);


void rope_neox(float* head, std::size_t head_dim, std::size_t pos, float theta);
void matvec(const Mat& m, const float* x, float* y, float* scratch);
void embed_row(const Mat& m, std::uint64_t token, float* out);

}  // namespace tinyinfer