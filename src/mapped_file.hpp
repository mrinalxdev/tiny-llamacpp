#pragma once
// Read-only memory-mapped file (POSIX).
//
// Why mmap instead of read()? A Q4_K_M 3B model is ~2 GB. With mmap the kernel
// pages weights in lazily and can evict them under memory pressure, so on an
// 8 GB machine we never hold a private second copy of the weights.

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tinyinfer {

class MappedFile {
public:
    MappedFile() = default;

    explicit MappedFile(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error("cannot open '" + path + "': " + std::strerror(errno));
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            int e = errno;
            ::close(fd);
            throw std::runtime_error("fstat '" + path + "': " + std::strerror(e));
        }
        if (st.st_size <= 0) {
            ::close(fd);
            throw std::runtime_error("file is empty: '" + path + "'");
        }
        size_ = static_cast<std::size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        int e = errno;
        ::close(fd);  // the mapping stays valid after close
        if (p == MAP_FAILED) {
            size_ = 0;
            throw std::runtime_error("mmap '" + path + "': " + std::strerror(e));
        }
        data_ = static_cast<const std::byte*>(p);
    }

    ~MappedFile() { reset(); }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}

    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) {
            reset();
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }
    std::size_t size() const noexcept { return size_; }

private:
    void reset() noexcept {
        if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace tinyinfer
