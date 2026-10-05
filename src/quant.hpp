#pragma once

#include <cstddef>
#include <cstdint>

#include "gguf.hpp"

namespace tinyinfer {

constexpr int QK_K = 256;

struct BlockQ4K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};
struct BlockQ6K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};

void dequantize_row(GgmlType type, const void* src, float* dst, std::size_t n);

void dequantize_row_q4_k(const BlockQ4K* src, float* dst, std::size_t n);
void dequantize_row_q6_k(const BlockQ6K* src, float* dst, std::size_t n);

}
