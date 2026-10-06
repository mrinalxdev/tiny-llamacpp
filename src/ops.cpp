#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <vector>

#include "quant.hpp"

namespace tinyinfer {

Mat Mat::from(const Gguf& g, std::string_view name) {
    const TensorInfo& t = g.tensor(name);
    if (t.n_dims != 2) {
        throw GgufError(std::format("tensor '{}' is {}-D, expected a matrix", name, t.n_dims));
    }
    Mat m;
    m.type = t.type;
    m.data = g.tensor_data(t).data();
    m.cols = t.ne[0];
    m.rows = t.ne[1];
    m.row_bytes = t.row_bytes;
    return m;
}

void rmsnorm(float* out, const float* x, const float* w, std::size_t n, float eps) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float scale = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

void softmax(float* x, std::size_t n) {
    if (n == 0) return;
    const float mx = *std::max_element(x, x + n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

void silu_mul(float* gate, const float* up, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) gate[i] = silu(gate[i]) * up[i];
}

void add_inplace(float* out, const float* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] += x[i];
}

float dot(const float* a, const float* b, std::size_t n) {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void rope_neox(float* head, std::size_t head_dim, std::size_t pos, float theta) {
    const std::size_t half = head_dim / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const double freq = std::pow(static_cast<double>(theta), -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
        const double angle = static_cast<double>(pos) * freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        const float a = head[i];
        const float b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
}

void matvec(const Mat& m, const float* x, float* y, float* scratch) {
    for (std::uint64_t r = 0; r < m.rows; ++r) {
        dequantize_row(m.type, m.row(r), scratch, static_cast<std::size_t>(m.cols));
        y[r] = dot(scratch, x, static_cast<std::size_t>(m.cols));
    }
}

void embed_row(const Mat& m, std::uint64_t token, float* out) {
    if (token >= m.rows) throw std::out_of_range(std::format("token id {} >= vocab {}", token, m.rows));
    dequantize_row(m.type, m.row(token), out, static_cast<std::size_t>(m.cols));
}

}  // namespace tinyinfer