#include "quant.hpp"

#include <cstring>
#include <stdexcept>

namespace tinyinfer {

static_assert(sizeof(BlockQ4K) == 144, "Q4_K block must be 144 bytes");
static_assert(sizeof(BlockQ6K) == 210, "Q6_K block must be 210 bytes");

namespace {

inline void get_scale_min(int j, const std::uint8_t* q, std::uint8_t& sc, std::uint8_t& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

}

void dequantize_row_q4_k(const BlockQ4K* x, float* y, std::size_t n) {
    if (n % QK_K) throw std::invalid_argument("Q4_K row length must be a multiple of 256");
    const std::size_t nb = n / QK_K;
    for (std::size_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const std::uint8_t* q = x[i].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            std::uint8_t sc, mn;
            get_scale_min(is + 0, x[i].scales, sc, mn);
            const float d1 = d * sc, m1 = dmin * mn;
            get_scale_min(is + 1, x[i].scales, sc, mn);
            const float d2 = d * sc, m2 = dmin * mn;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q6_k(const BlockQ6K* x, float* y, std::size_t n) {
    if (n % QK_K) throw std::invalid_argument("Q6_K row length must be a multiple of 256");
    const std::size_t nb = n / QK_K;
    for (std::size_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        const std::uint8_t* ql = x[i].ql;
        const std::uint8_t* qh = x[i].qh;
        const std::int8_t* sc = x[i].scales;
        for (int half = 0; half < QK_K; half += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row(GgmlType type, const void* src, float* dst, std::size_t n) {
    /*
     * Dequantizes one row of n weights into dst without expanding the whole model.
     * F32 is copied as-is. F16 and BF16 are converted element-wise (memcpy used
     * because the source may be unaligned). Q4_K reconstructs each value as
     * scale * q - min from 256-value blocks with bit-packed 6-bit scales/mins.
     * Q6_K reconstructs each value as scale * q, with 6-bit quants re-centered
     * by -32. Unsupported types throw std::invalid_argument.
     */
    switch (type) {
        case GgmlType::F32:
            std::memcpy(dst, src, n * sizeof(float));
            return;
        case GgmlType::F16: {
            const auto* p = static_cast<const std::uint8_t*>(src);
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t h;
                std::memcpy(&h, p + 2 * i, 2);
                dst[i] = f16_to_f32(h);
            }
            return;
        }
        case GgmlType::BF16: {
            const auto* p = static_cast<const std::uint8_t*>(src);
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t h;
                std::memcpy(&h, p + 2 * i, 2);
                dst[i] = bf16_to_f32(h);
            }
            return;
        }
        case GgmlType::Q4_K:
            dequantize_row_q4_k(static_cast<const BlockQ4K*>(src), dst, n);
            return;
        case GgmlType::Q6_K:
            dequantize_row_q6_k(static_cast<const BlockQ6K*>(src), dst, n);
            return;
        default:
            throw std::invalid_argument(std::string("dequantize_row: unsupported type ") +
                                        (type_info(type) ? type_info(type)->name : "?"));
    }
}

}
