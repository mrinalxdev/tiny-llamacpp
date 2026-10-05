#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "mapped_file.hpp"

namespace tinyinfer {

static_assert(std::endian::native == std::endian::little,
              "TinyInfer assumes a little-endian host (x86-64 / aarch64).");

struct GgufError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class MetaType : std::uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
    String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
};

const char* meta_type_name(MetaType t) noexcept;

struct MetaValue;

struct MetaArray {
    MetaType elem_type{};
    std::vector<MetaValue> items;
};

struct MetaValue {
    using Variant = std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                                 std::uint32_t, std::int32_t, float, bool, std::string,
                                 std::uint64_t, std::int64_t, double, MetaArray>;
    Variant v;
};

enum class GgmlType : std::uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, BF16 = 30,
};

struct TypeInfo {
    const char* name;
    std::uint32_t block_elems;
    std::uint32_t block_bytes;
};

const TypeInfo* type_info(GgmlType t) noexcept;
const TypeInfo* type_info(std::uint32_t raw_type) noexcept;

struct TensorInfo {
    std::string name;
    std::uint32_t n_dims = 0;
    std::array<std::uint64_t, 4> ne{1, 1, 1, 1};  // unused dims are 1
    GgmlType type = GgmlType::F32;
    std::uint64_t offset = 0;
    std::uint64_t n_elements = 0;
    std::uint64_t nbytes = 0;
    std::uint64_t row_bytes = 0;

    std::uint64_t n_rows() const { return ne[1] * ne[2] * ne[3]; }
};

// ------------------------------------------------------------------- Gguf ---

class Gguf {
public:
    static Gguf open(const std::string& path);

    std::uint32_t version() const noexcept { return version_; }
    std::uint64_t file_size() const noexcept { return file_.size(); }
    std::uint64_t alignment() const noexcept { return alignment_; }
    std::uint64_t data_offset() const noexcept { return data_offset_; }

    const std::vector<std::pair<std::string, MetaValue>>& metadata() const noexcept { return meta_; }
    const MetaValue* find_meta(std::string_view key) const;
    std::optional<std::uint64_t> get_uint(std::string_view key) const;
    std::optional<double> get_float(std::string_view key) const;
    std::optional<std::string_view> get_string(std::string_view key) const;

    const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
    const TensorInfo* find_tensor(std::string_view name) const;
    const TensorInfo& tensor(std::string_view name) const;  // throws if missing
    std::span<const std::byte> tensor_data(const TensorInfo& t) const;

private:
    Gguf() = default;

    struct StrHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };
    template <class V>
    using StrMap = std::unordered_map<std::string, V, StrHash, std::equal_to<>>;

    MappedFile file_;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = 32;
    std::uint64_t data_offset_ = 0;
    std::vector<std::pair<std::string, MetaValue>> meta_;
    StrMap<std::size_t> meta_index_;
    std::vector<TensorInfo> tensors_;
    StrMap<std::size_t> tensor_index_;
};

inline float bf16_to_f32(std::uint16_t h) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
}

inline float f16_to_f32(std::uint16_t h) noexcept {
    std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +-0
        } else {
            int e = -1;
            do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}

}  // namespace tinyinfer
