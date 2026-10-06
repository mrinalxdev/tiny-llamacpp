#include "gguf.hpp"

#include <cstring>
#include <format>
#include <type_traits>

namespace tinyinfer {

// ============================================================ type tables ===

const char* meta_type_name(MetaType t) noexcept {
    switch (t) {
        case MetaType::U8: return "u8";
        case MetaType::I8: return "i8";
        case MetaType::U16: return "u16";
        case MetaType::I16: return "i16";
        case MetaType::U32: return "u32";
        case MetaType::I32: return "i32";
        case MetaType::F32: return "f32";
        case MetaType::Bool: return "bool";
        case MetaType::String: return "string";
        case MetaType::Array: return "array";
        case MetaType::U64: return "u64";
        case MetaType::I64: return "i64";
        case MetaType::F64: return "f64";
    }
    return "?";
}

const TypeInfo* type_info(GgmlType t) noexcept {
    static constexpr TypeInfo kF32{"F32", 1, 4};
    static constexpr TypeInfo kF16{"F16", 1, 2};
    static constexpr TypeInfo kBF16{"BF16", 1, 2};
    static constexpr TypeInfo kF64{"F64", 1, 8};
    static constexpr TypeInfo kI8{"I8", 1, 1};
    static constexpr TypeInfo kI16{"I16", 1, 2};
    static constexpr TypeInfo kI32{"I32", 1, 4};
    static constexpr TypeInfo kI64{"I64", 1, 8};
    static constexpr TypeInfo kQ4_0{"Q4_0", 32, 18};
    static constexpr TypeInfo kQ4_1{"Q4_1", 32, 20};
    static constexpr TypeInfo kQ5_0{"Q5_0", 32, 22};
    static constexpr TypeInfo kQ5_1{"Q5_1", 32, 24};
    static constexpr TypeInfo kQ8_0{"Q8_0", 32, 34};
    static constexpr TypeInfo kQ8_1{"Q8_1", 32, 36};
    static constexpr TypeInfo kQ2_K{"Q2_K", 256, 84};
    static constexpr TypeInfo kQ3_K{"Q3_K", 256, 110};
    static constexpr TypeInfo kQ4_K{"Q4_K", 256, 144};
    static constexpr TypeInfo kQ5_K{"Q5_K", 256, 176};
    static constexpr TypeInfo kQ6_K{"Q6_K", 256, 210};
    static constexpr TypeInfo kQ8_K{"Q8_K", 256, 292};
    switch (t) {
        case GgmlType::F32: return &kF32;
        case GgmlType::F16: return &kF16;
        case GgmlType::BF16: return &kBF16;
        case GgmlType::F64: return &kF64;
        case GgmlType::I8: return &kI8;
        case GgmlType::I16: return &kI16;
        case GgmlType::I32: return &kI32;
        case GgmlType::I64: return &kI64;
        case GgmlType::Q4_0: return &kQ4_0;
        case GgmlType::Q4_1: return &kQ4_1;
        case GgmlType::Q5_0: return &kQ5_0;
        case GgmlType::Q5_1: return &kQ5_1;
        case GgmlType::Q8_0: return &kQ8_0;
        case GgmlType::Q8_1: return &kQ8_1;
        case GgmlType::Q2_K: return &kQ2_K;
        case GgmlType::Q3_K: return &kQ3_K;
        case GgmlType::Q4_K: return &kQ4_K;
        case GgmlType::Q5_K: return &kQ5_K;
        case GgmlType::Q6_K: return &kQ6_K;
        case GgmlType::Q8_K: return &kQ8_K;
    }
    return nullptr;
}

const TypeInfo* type_info(std::uint32_t raw) noexcept {
    return type_info(static_cast<GgmlType>(raw));
}

namespace {

class Reader {
public:
    explicit Reader(std::span<const std::byte> d) : data_(d) {}

    std::size_t pos() const noexcept { return pos_; }
    std::size_t remaining() const noexcept { return data_.size() - pos_; }

    void require(std::uint64_t n) const {
        if (n > remaining()) {
            throw GgufError(std::format("unexpected end of file at offset {}: need {} bytes, {} left",
                                        pos_, n, remaining()));
        }
    }

    template <class T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(sizeof(T));
        T v;
        std::memcpy(&v, data_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string read_string() {
        std::uint64_t n = read<std::uint64_t>();
        require(n);
        std::string s(reinterpret_cast<const char*>(data_.data() + pos_), static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return s;
    }

    void seek(std::size_t p) {
        if (p > data_.size()) throw GgufError("seek past end of file");
        pos_ = p;
    }

private:
    std::span<const std::byte> data_;
    std::size_t pos_ = 0;
};

template <class T>
MetaValue make(T x) {
    return MetaValue{MetaValue::Variant(std::in_place_type<T>, std::move(x))};
}

bool valid_meta_type(std::uint32_t t) { return t <= static_cast<std::uint32_t>(MetaType::F64); }

MetaValue read_value(Reader& r, MetaType t, int depth) {
    switch (t) {
        case MetaType::U8: return make(r.read<std::uint8_t>());
        case MetaType::I8: return make(r.read<std::int8_t>());
        case MetaType::U16: return make(r.read<std::uint16_t>());
        case MetaType::I16: return make(r.read<std::int16_t>());
        case MetaType::U32: return make(r.read<std::uint32_t>());
        case MetaType::I32: return make(r.read<std::int32_t>());
        case MetaType::F32: return make(r.read<float>());
        case MetaType::Bool: return make(r.read<std::uint8_t>() != 0);
        case MetaType::String: return make(r.read_string());
        case MetaType::U64: return make(r.read<std::uint64_t>());
        case MetaType::I64: return make(r.read<std::int64_t>());
        case MetaType::F64: return make(r.read<double>());
        case MetaType::Array: {
            if (depth >= 4) throw GgufError("metadata arrays nested too deeply");
            std::uint32_t et_raw = r.read<std::uint32_t>();
            if (!valid_meta_type(et_raw)) {
                throw GgufError(std::format("bad array element type {} at offset {}", et_raw, r.pos()));
            }
            std::uint64_t count = r.read<std::uint64_t>();          
            if (count > r.remaining()) throw GgufError("array length exceeds file size");
            MetaArray arr;
            arr.elem_type = static_cast<MetaType>(et_raw);
            arr.items.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                arr.items.push_back(read_value(r, arr.elem_type, depth + 1));
            }
            return make(std::move(arr));
        }
    }
    throw GgufError("unreachable metadata type");
}

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace


Gguf Gguf::open(const std::string& path) {
    Gguf g;
    g.file_ = MappedFile(path);
    Reader r(g.file_.bytes());

    constexpr std::uint32_t kMagic = 0x46554747;  // "GGUF" read as little-endian u32
    if (r.read<std::uint32_t>() != kMagic) {
        throw GgufError("not a GGUF file (bad magic)");
    }
    g.version_ = r.read<std::uint32_t>();
    if (g.version_ != 2 && g.version_ != 3) {
        throw GgufError(std::format("unsupported GGUF version {} (supported: 2, 3)", g.version_));
    }
    const std::uint64_t tensor_count = r.read<std::uint64_t>();
    const std::uint64_t kv_count = r.read<std::uint64_t>();
    if (tensor_count > r.remaining() || kv_count > r.remaining()) {
        throw GgufError("tensor/metadata count exceeds file size (corrupt header)");
    }

    g.meta_.reserve(static_cast<std::size_t>(kv_count));
    for (std::uint64_t i = 0; i < kv_count; ++i) {
        std::string key = r.read_string();
        std::uint32_t type_raw = r.read<std::uint32_t>();
        if (!valid_meta_type(type_raw)) {
            throw GgufError(std::format("key '{}': bad value type {}", key, type_raw));
        }
        MetaValue val = read_value(r, static_cast<MetaType>(type_raw), 0);
        if (g.meta_index_.contains(key)) throw GgufError("duplicate metadata key '" + key + "'");
        g.meta_index_.emplace(key, g.meta_.size());
        g.meta_.emplace_back(std::move(key), std::move(val));
    }

    if (auto a = g.get_uint("general.alignment")) g.alignment_ = *a;
    if (g.alignment_ == 0 || !std::has_single_bit(g.alignment_)) {
        throw GgufError(std::format("invalid general.alignment {}", g.alignment_));
    }

    // ---- tensor table -----------------------------------------------------
    g.tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = r.read_string();
        t.n_dims = r.read<std::uint32_t>();
        if (t.n_dims == 0 || t.n_dims > 4) {
            throw GgufError(std::format("tensor '{}': unsupported n_dims {}", t.name, t.n_dims));
        }
        t.n_elements = 1;
        for (std::uint32_t d = 0; d < t.n_dims; ++d) {
            t.ne[d] = r.read<std::uint64_t>();
            if (t.ne[d] == 0) throw GgufError("tensor '" + t.name + "': zero-sized dimension");
            if (__builtin_mul_overflow(t.n_elements, t.ne[d], &t.n_elements)) {
                throw GgufError("tensor '" + t.name + "': element count overflow");
            }
        }
        std::uint32_t type_raw = r.read<std::uint32_t>();
        const TypeInfo* ti = type_info(type_raw);
        if (!ti) {
            throw GgufError(std::format("tensor '{}': unsupported ggml type {}", t.name, type_raw));
        }
        t.type = static_cast<GgmlType>(type_raw);
        t.offset = r.read<std::uint64_t>();
        if (t.ne[0] % ti->block_elems != 0) {
            throw GgufError(std::format("tensor '{}': ne[0]={} not divisible by block size {} ({})",
                                        t.name, t.ne[0], ti->block_elems, ti->name));
        }
        t.row_bytes = t.ne[0] / ti->block_elems * ti->block_bytes;
        t.nbytes = t.row_bytes * t.n_rows();

        if (t.offset % g.alignment_ != 0) {
            throw GgufError("tensor '" + t.name + "': offset not aligned");
        }
        if (g.tensor_index_.contains(t.name)) throw GgufError("duplicate tensor '" + t.name + "'");
        g.tensor_index_.emplace(t.name, g.tensors_.size());
        g.tensors_.push_back(std::move(t));
    }

    // ---- data section -----------------------------------------------------
    g.data_offset_ = align_up(r.pos(), g.alignment_);
    const std::uint64_t fsize = g.file_.size();
    if (g.data_offset_ > fsize) throw GgufError("file ends before the tensor data section");
    const std::uint64_t avail = fsize - g.data_offset_;
    for (const auto& t : g.tensors_) {
        if (t.offset > avail || t.nbytes > avail - t.offset) {
            throw GgufError(std::format("tensor '{}' (offset {}, {} bytes) extends past end of file "
                                        "(truncated download?)",
                                        t.name, t.offset, t.nbytes));
        }
    }
    return g;
}

// ===================================================i=========== accessors ===

const MetaValue* Gguf::find_meta(std::string_view key) const {
    auto it = meta_index_.find(key);
    return it == meta_index_.end() ? nullptr : &meta_[it->second].second;
}

std::optional<std::uint64_t> Gguf::get_uint(std::string_view key) const {
    const MetaValue* mv = find_meta(key);
    if (!mv) return std::nullopt;
    return std::visit(
        [](const auto& x) -> std::optional<std::uint64_t> {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                if constexpr (std::is_signed_v<T>) {
                    if (x < 0) return std::nullopt;
                }
                return static_cast<std::uint64_t>(x);
            } else {
                return std::nullopt;
            }
        },
        mv->v);
}

std::optional<double> Gguf::get_float(std::string_view key) const {
    const MetaValue* mv = find_meta(key);
    if (!mv) return std::nullopt;
    return std::visit(
        [](const auto& x) -> std::optional<double> {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
                return static_cast<double>(x);
            } else {
                return std::nullopt;
            }
        },
        mv->v);
}

std::optional<std::string_view> Gguf::get_string(std::string_view key) const {
    const MetaValue* mv = find_meta(key);
    if (!mv) return std::nullopt;
    if (const auto* s = std::get_if<std::string>(&mv->v)) return std::string_view(*s);
    return std::nullopt;
}

const TensorInfo* Gguf::find_tensor(std::string_view name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& Gguf::tensor(std::string_view name) const {
    if (const TensorInfo* t = find_tensor(name)) return *t;
    throw GgufError("tensor not found: " + std::string(name));
}

std::span<const std::byte> Gguf::tensor_data(const TensorInfo& t) const {
    // Bounds were validated in open(); this is pure pointer arithmetic.
    return file_.bytes().subspan(static_cast<std::size_t>(data_offset_ + t.offset),
                                 static_cast<std::size_t>(t.nbytes));
}

}  // namespace tinyinfer
