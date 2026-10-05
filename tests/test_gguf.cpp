#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gguf.hpp"

using namespace tinyinfer;

static int g_failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                           \
        }                                                                           \
    } while (0)

#define CHECK_THROWS(expr)                                                          \
    do {                                                                            \
        bool threw = false;                                                         \
        try { expr; } catch (const std::exception&) { threw = true; }               \
        if (!threw) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: expected exception: %s\n", __FILE__, __LINE__, #expr); \
            ++g_failures;                                                           \
        }                                                                           \
    } while (0)

// ------------------------------------------------------------ file builder --
struct Buf {
    std::vector<unsigned char> b;
    template <class T> void put(T v) {
        unsigned char tmp[sizeof(T)];
        std::memcpy(tmp, &v, sizeof(T));
        b.insert(b.end(), tmp, tmp + sizeof(T));
    }
    void str(const std::string& s) {
        put<std::uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void pad_to(std::size_t a) { while (b.size() % a) b.push_back(0); }
};

static std::vector<unsigned char> build_sample() {
    Buf f;
    f.put<std::uint32_t>(0x46554747);
    f.put<std::uint32_t>(3);
    f.put<std::uint64_t>(2);  // tensors
    f.put<std::uint64_t>(5);  // kvs

    f.str("general.architecture"); f.put<std::uint32_t>(8); f.str("test");
    f.str("test.block_count");     f.put<std::uint32_t>(4); f.put<std::uint32_t>(2);
    f.str("test.rope.freq_base");  f.put<std::uint32_t>(6); f.put<float>(1000000.0f);
    f.str("test.flag");            f.put<std::uint32_t>(7); f.put<std::uint8_t>(1);
    f.str("test.names");           f.put<std::uint32_t>(9); f.put<std::uint32_t>(8); f.put<std::uint64_t>(2);
    f.str("a"); f.str("bc");

    // tensor "w": F32 [4, 2] (8 floats = 32 bytes) at offset 0
    f.str("w"); f.put<std::uint32_t>(2); f.put<std::uint64_t>(4); f.put<std::uint64_t>(2);
    f.put<std::uint32_t>(0); f.put<std::uint64_t>(0);
    // tensor "h": F16 [8] (16 bytes) at offset 32
    f.str("h"); f.put<std::uint32_t>(1); f.put<std::uint64_t>(8);
    f.put<std::uint32_t>(1); f.put<std::uint64_t>(32);

    f.pad_to(32);  // start of data section
    for (int i = 0; i < 8; ++i) f.put<float>(static_cast<float>(i) + 0.5f);
    const std::uint16_t halves[8] = {0x3C00 /*1*/, 0xC000 /*-2*/, 0x3800 /*0.5*/, 0x0000,
                                     0x0001 /*2^-24*/, 0x7C00 /*inf*/, 0x4900 /*10*/, 0xBC00 /*-1*/};
    for (auto h : halves) f.put<std::uint16_t>(h);
    return f.b;
}

static std::string write_tmp(const std::string& name, const std::vector<unsigned char>& bytes, std::size_t len) {
    auto p = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(len));
    return p;
}

int main() {
    auto bytes = build_sample();

    // ---- happy path ----
    {
        Gguf g = Gguf::open(write_tmp("ti_ok.gguf", bytes, bytes.size()));
        CHECK(g.version() == 3);
        CHECK(g.tensors().size() == 2);
        CHECK(g.metadata().size() == 5);
        CHECK(g.alignment() == 32);
        CHECK(g.data_offset() % 32 == 0);
        CHECK(g.get_string("general.architecture") == "test");
        CHECK(g.get_uint("test.block_count") == 2u);
        CHECK(g.get_float("test.rope.freq_base") == 1000000.0);
        CHECK(!g.get_uint("test.rope.freq_base"));   // float is not an integer
        CHECK(!g.get_uint("does.not.exist"));
        const auto* names = std::get_if<MetaArray>(&g.find_meta("test.names")->v);
        CHECK(names && names->items.size() == 2);
        CHECK(names && std::get<std::string>(names->items[1].v) == "bc");

        const TensorInfo& w = g.tensor("w");
        CHECK(w.n_dims == 2 && w.ne[0] == 4 && w.ne[1] == 2);
        CHECK(w.nbytes == 32 && w.row_bytes == 16 && w.n_rows() == 2);
        auto wd = g.tensor_data(w);
        float f[8];
        std::memcpy(f, wd.data(), sizeof f);
        CHECK(f[0] == 0.5f && f[7] == 7.5f);

        const TensorInfo& h = g.tensor("h");
        auto hd = g.tensor_data(h);
        std::uint16_t u[8];
        std::memcpy(u, hd.data(), sizeof u);
        CHECK(f16_to_f32(u[0]) == 1.0f);
        CHECK(f16_to_f32(u[1]) == -2.0f);
        CHECK(f16_to_f32(u[2]) == 0.5f);
        CHECK(f16_to_f32(u[3]) == 0.0f);
        CHECK(std::fabs(f16_to_f32(u[4]) - std::ldexp(1.0f, -24)) == 0.0f);  // subnormal
        CHECK(std::isinf(f16_to_f32(u[5])));
        CHECK(f16_to_f32(u[6]) == 10.0f);
        CHECK(f16_to_f32(u[7]) == -1.0f);
        CHECK(g.find_tensor("nope") == nullptr);
        CHECK_THROWS(g.tensor("nope"));
    }

    // ---- bf16 ----
    CHECK(bf16_to_f32(0x3F80) == 1.0f);

    // ---- type table sanity (bits per weight) ----
    CHECK(type_info(GgmlType::Q4_K)->block_bytes * 8.0 / type_info(GgmlType::Q4_K)->block_elems == 4.5);
    CHECK(type_info(GgmlType::Q6_K)->block_bytes * 8.0 / type_info(GgmlType::Q6_K)->block_elems == 6.5625);
    CHECK(type_info(999u) == nullptr);

    // ---- malformed inputs must throw, never crash ----
    {
        auto bad = bytes; bad[0] = 'X';
        CHECK_THROWS(Gguf::open(write_tmp("ti_magic.gguf", bad, bad.size())));
    }
    {
        auto bad = bytes; bad[4] = 9;  // version 9
        CHECK_THROWS(Gguf::open(write_tmp("ti_ver.gguf", bad, bad.size())));
    }
    {
        auto bad = bytes;
        for (int i = 0; i < 8; ++i) bad[8 + i] = 0xFF;  // absurd tensor_count
        CHECK_THROWS(Gguf::open(write_tmp("ti_count.gguf", bad, bad.size())));
    }
    // Truncate at every length: the loader must throw or succeed, never read out of bounds.
    // (Only the full file is valid; everything shorter must throw.)
    for (std::size_t len = 1; len < bytes.size(); ++len) {
        CHECK_THROWS(Gguf::open(write_tmp("ti_trunc.gguf", bytes, len)));
    }
    CHECK_THROWS(Gguf::open("/nonexistent/path/model.gguf"));

    if (g_failures == 0) std::puts("all GGUF tests passed");
    return g_failures ? 1 : 0;
}
