// tinyinfer-inspect: dump what is inside a GGUF file.
//
//   tinyinfer-inspect model.gguf                  summary + layer-0 tensors
//   tinyinfer-inspect model.gguf --kv             all metadata (long values truncated)
//   tinyinfer-inspect model.gguf --tensors        every tensor
//   tinyinfer-inspect model.gguf --dump NAME [N]  first N values of a tensor (F32/F16/BF16)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <unordered_set>

#include "gguf.hpp"

using namespace tinyinfer;

namespace {

std::string human_bytes(std::uint64_t n) {
    constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(n);
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    return u == 0 ? std::format("{} B", n) : std::format("{:.2f} {}", v, units[u]);
}

std::string shape_str(const TensorInfo& t) {
    std::string s = "[";
    for (std::uint32_t i = 0; i < t.n_dims; ++i) {
        if (i) s += ", ";
        s += std::to_string(t.ne[i]);
    }
    return s + "]";
}

std::string escape_trunc(const std::string& in, std::size_t max_len) {
    std::string out;
    for (char c : in) {
        if (out.size() >= max_len) { out += "..."; break; }
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

std::string preview(const MetaValue& mv) {
    return std::visit(
        [](const auto& x) -> std::string {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return "\"" + escape_trunc(x, 100) + "\"";
            } else if constexpr (std::is_same_v<T, bool>) {
                return x ? "true" : "false";
            } else if constexpr (std::is_same_v<T, MetaArray>) {
                std::string s = std::format("array<{}> len={} [", meta_type_name(x.elem_type), x.items.size());
                for (std::size_t i = 0; i < x.items.size() && i < 4; ++i) {
                    if (i) s += ", ";
                    s += preview(x.items[i]);
                }
                if (x.items.size() > 4) s += ", ...";
                return s + "]";
            } else if constexpr (std::is_integral_v<T>) {
                return std::format("{}", static_cast<long long>(x) );
            } else {
                return std::format("{}", x);
            }
        },
        mv.v);
}

const char* meta_value_type_name(const MetaValue& mv) {
    static constexpr MetaType map[] = {MetaType::U8, MetaType::I8, MetaType::U16, MetaType::I16,
                                       MetaType::U32, MetaType::I32, MetaType::F32, MetaType::Bool,
                                       MetaType::String, MetaType::U64, MetaType::I64, MetaType::F64,
                                       MetaType::Array};
    return meta_type_name(map[mv.v.index()]);
}

void print_file_info(const Gguf& g, const std::string& path) {
    std::puts("== File ==");
    std::printf("  path:          %s\n", path.c_str());
    std::printf("  size:          %s\n", human_bytes(g.file_size()).c_str());
    std::printf("  GGUF version:  %u\n", g.version());
    std::printf("  metadata keys: %zu\n", g.metadata().size());
    std::printf("  tensors:       %zu\n", g.tensors().size());
    std::printf("  alignment:     %llu\n", static_cast<unsigned long long>(g.alignment()));
    std::printf("  data offset:   %llu\n\n", static_cast<unsigned long long>(g.data_offset()));
}

void print_summary(const Gguf& g) {
    std::puts("== Model summary ==");
    auto arch_opt = g.get_string("general.architecture");
    std::string arch = arch_opt ? std::string(*arch_opt) : "?";
    auto s = [&](const char* label, std::string_view key) {
        if (auto v = g.get_string(key)) std::printf("  %-24s %s\n", label, std::string(*v).c_str());
    };
    auto u = [&](const char* label, const std::string& key) {
        if (auto v = g.get_uint(key)) std::printf("  %-24s %llu\n", label, static_cast<unsigned long long>(*v));
    };
    auto f = [&](const char* label, const std::string& key) {
        if (auto v = g.get_float(key)) std::printf("  %-24s %g\n", label, *v);
    };
    s("name", "general.name");
    s("architecture", "general.architecture");
    if (auto ft = g.get_uint("general.file_type"))
        std::printf("  %-24s %llu (llama.cpp ftype id; 15 = Q4_K_M)\n", "file_type", static_cast<unsigned long long>(*ft));
    u("context_length", arch + ".context_length");
    u("embedding_length", arch + ".embedding_length");
    u("block_count", arch + ".block_count");
    u("feed_forward_length", arch + ".feed_forward_length");
    u("attention.head_count", arch + ".attention.head_count");
    u("attention.head_count_kv", arch + ".attention.head_count_kv");
    f("rope.freq_base", arch + ".rope.freq_base");
    f("rms_norm_eps", arch + ".attention.layer_norm_rms_epsilon");
    s("tokenizer.model", "tokenizer.ggml.model");
    s("tokenizer.pre", "tokenizer.ggml.pre");
    if (const MetaValue* toks = g.find_meta("tokenizer.ggml.tokens"))
        if (const auto* a = std::get_if<MetaArray>(&toks->v))
            std::printf("  %-24s %zu\n", "vocab size (tokens)", a->items.size());
    u("tokenizer.bos_token_id", "tokenizer.ggml.bos_token_id");
    u("tokenizer.eos_token_id", "tokenizer.ggml.eos_token_id");
    std::puts("");
}

void print_type_histogram(const Gguf& g) {
    struct Agg { std::size_t count = 0; std::uint64_t bytes = 0; std::uint64_t elems = 0; };
    std::map<std::string, Agg> agg;
    std::uint64_t total_bytes = 0, total_elems = 0;
    for (const auto& t : g.tensors()) {
        auto& a = agg[type_info(t.type)->name];
        a.count++; a.bytes += t.nbytes; a.elems += t.n_elements;
        total_bytes += t.nbytes; total_elems += t.n_elements;
    }
    std::puts("== Tensor types ==");
    for (const auto& [name, a] : agg) {
        std::printf("  %-6s %4zu tensors  %12s  %6.2f%% of bytes\n", name.c_str(), a.count,
                    human_bytes(a.bytes).c_str(), 100.0 * static_cast<double>(a.bytes) / static_cast<double>(total_bytes));
    }
    std::printf("  total: %s, %.3f B parameters, %.2f bits/weight\n\n", human_bytes(total_bytes).c_str(),
                static_cast<double>(total_elems) / 1e9,
                8.0 * static_cast<double>(total_bytes) / static_cast<double>(total_elems));
}

void print_tensor_line(const TensorInfo& t) {
    std::printf("  %-34s %-6s %-22s %12s  @%llu\n", t.name.c_str(), type_info(t.type)->name,
                shape_str(t).c_str(), human_bytes(t.nbytes).c_str(), static_cast<unsigned long long>(t.offset));
}

void print_tensors(const Gguf& g, bool all) {
    std::puts(all ? "== All tensors ==" : "== Tensors (non-block + blk.0; use --tensors for all) ==");
    for (const auto& t : g.tensors()) {
        if (all || !t.name.starts_with("blk.") || t.name.starts_with("blk.0.")) print_tensor_line(t);
    }
    std::puts("");
}

// Verifies that the file contains exactly the tensors, with exactly the shapes,
// that a Qwen2 (Qwen2.5) transformer needs. Returns number of problems.
int check_qwen2_layout(const Gguf& g) {
    std::puts("== Qwen2 layout check ==");
    auto arch = g.get_string("general.architecture");
    if (!arch || *arch != "qwen2") {
        std::printf("  architecture is '%s', not 'qwen2' - skipping.\n\n", arch ? std::string(*arch).c_str() : "?");
        return 0;
    }
    auto need = [&](const char* k) -> std::uint64_t {
        auto v = g.get_uint(k);
        if (!v) { std::printf("  PROBLEM: missing metadata %s\n", k); return 0; }
        return *v;
    };
    const std::uint64_t n_layer = need("qwen2.block_count");
    const std::uint64_t n_embd = need("qwen2.embedding_length");
    const std::uint64_t n_head = need("qwen2.attention.head_count");
    const std::uint64_t n_kv = need("qwen2.attention.head_count_kv");
    const std::uint64_t n_ff = need("qwen2.feed_forward_length");
    if (!n_layer || !n_embd || !n_head || !n_kv || !n_ff) { std::puts(""); return 1; }
    const std::uint64_t head_dim = n_embd / n_head;
    std::uint64_t n_vocab = 0;
    if (const MetaValue* toks = g.find_meta("tokenizer.ggml.tokens"))
        if (const auto* a = std::get_if<MetaArray>(&toks->v)) n_vocab = a->items.size();

    std::printf("  derived: head_dim=%llu  q_dim=%llu  kv_dim=%llu  gqa_group=%llu  vocab=%llu\n",
                (unsigned long long)head_dim, (unsigned long long)(n_head * head_dim),
                (unsigned long long)(n_kv * head_dim), (unsigned long long)(n_head / n_kv),
                (unsigned long long)n_vocab);

    int problems = 0;
    std::unordered_set<std::string> expected;
    auto expect = [&](const std::string& name, std::initializer_list<std::uint64_t> dims) {
        expected.insert(name);
        const TensorInfo* t = g.find_tensor(name);
        if (!t) { std::printf("  PROBLEM: missing tensor %s\n", name.c_str()); ++problems; return; }
        bool ok = t->n_dims == dims.size() && std::equal(dims.begin(), dims.end(), t->ne.begin());
        if (!ok) {
            std::printf("  PROBLEM: %s has shape %s, expected %zu dims starting [", name.c_str(),
                        shape_str(*t).c_str(), dims.size());
            bool first = true;
            for (auto d : dims) { std::printf(first ? "%llu" : ", %llu", (unsigned long long)d); first = false; }
            std::puts("]");
            ++problems;
        }
    };

    expect("token_embd.weight", {n_embd, n_vocab ? n_vocab : g.tensor("token_embd.weight").ne[1]});
    expect("output_norm.weight", {n_embd});
    const bool tied = g.find_tensor("output.weight") == nullptr;
    if (!tied) expect("output.weight", {n_embd, n_vocab ? n_vocab : g.tensor("token_embd.weight").ne[1]});

    const std::uint64_t q_dim = n_head * head_dim, kv_dim = n_kv * head_dim;
    for (std::uint64_t i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        expect(p + "attn_norm.weight", {n_embd});
        expect(p + "attn_q.weight", {n_embd, q_dim});
        expect(p + "attn_q.bias", {q_dim});
        expect(p + "attn_k.weight", {n_embd, kv_dim});
        expect(p + "attn_k.bias", {kv_dim});
        expect(p + "attn_v.weight", {n_embd, kv_dim});
        expect(p + "attn_v.bias", {kv_dim});
        expect(p + "attn_output.weight", {q_dim, n_embd});
        expect(p + "ffn_norm.weight", {n_embd});
        expect(p + "ffn_gate.weight", {n_embd, n_ff});
        expect(p + "ffn_up.weight", {n_embd, n_ff});
        expect(p + "ffn_down.weight", {n_ff, n_embd});
    }
    for (const auto& t : g.tensors()) {
        if (!expected.contains(t.name)) {
            std::printf("  NOTE: unexpected tensor %s %s\n", t.name.c_str(), shape_str(t).c_str());
        }
    }
    std::printf("  output projection: %s\n", tied ? "TIED to token_embd.weight (no output.weight in file)" : "separate output.weight");
    std::printf("  result: %s\n\n", problems ? "PROBLEMS FOUND" : "OK - all expected tensors present with correct shapes");
    return problems;
}

void print_kv(const Gguf& g) {
    std::puts("== Metadata ==");
    for (const auto& [k, v] : g.metadata()) {
        std::printf("  %-44s %-7s %s\n", k.c_str(), meta_value_type_name(v), preview(v).c_str());
    }
    std::puts("");
}

int dump_tensor(const Gguf& g, const std::string& name, std::size_t n) {
    const TensorInfo* t = g.find_tensor(name);
    if (!t) { std::fprintf(stderr, "no such tensor: %s\n", name.c_str()); return 1; }
    auto data = g.tensor_data(*t);
    n = std::min<std::size_t>(n, static_cast<std::size_t>(t->ne[0]));
    std::printf("%s  type=%s shape=%s  first %zu values of row 0:\n  ", name.c_str(), type_info(t->type)->name,
                shape_str(*t).c_str(), n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = 0;
        switch (t->type) {
            case GgmlType::F32: std::memcpy(&v, data.data() + i * 4, 4); break;
            case GgmlType::F16: { std::uint16_t h; std::memcpy(&h, data.data() + i * 2, 2); v = f16_to_f32(h); break; }
            case GgmlType::BF16: { std::uint16_t h; std::memcpy(&h, data.data() + i * 2, 2); v = bf16_to_f32(h); break; }
            default: {
                std::printf("(quantized type - dequantization comes in Step 2) raw bytes: ");
                for (std::size_t b = 0; b < std::min<std::size_t>(32, data.size()); ++b)
                    std::printf("%02x ", static_cast<unsigned>(data[b]));
                std::puts("");
                return 0;
            }
        }
        std::printf("%.6g ", v);
    }
    std::puts("");
    return 0;
}

void usage() {
    std::fputs("usage: tinyinfer-inspect <model.gguf> [--kv] [--tensors] [--dump NAME [N]]\n", stderr);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    std::string path = argv[1];
    bool want_kv = false, want_all_tensors = false;
    std::string dump_name;
    std::size_t dump_n = 8;
    for (int i = 2; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--kv") want_kv = true;
        else if (a == "--tensors") want_all_tensors = true;
        else if (a == "--dump" && i + 1 < argc) {
            dump_name = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-') dump_n = std::strtoull(argv[++i], nullptr, 10);
        } else { usage(); return 2; }
    }

    try {
        Gguf g = Gguf::open(path);
        print_file_info(g, path);
        print_summary(g);
        print_type_histogram(g);
        print_tensors(g, want_all_tensors);
        if (want_kv) print_kv(g);
        int problems = check_qwen2_layout(g);
        if (!dump_name.empty()) {
            if (dump_tensor(g, dump_name, dump_n) != 0) return 1;
        }
        return problems ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
