#include "Serialization.h"

#include <nlohmann/json.hpp>
#include <mio/mmap.hpp>


using json = nlohmann::json;
using spdlog::fmt_lib::format;

class SafeTensors::MMapImpl {
public:
    virtual ~MMapImpl() {}
    virtual size_t size() = 0;
    virtual const char *data() = 0;
};

class SafeTensors::MMapImplMio : public SafeTensors::MMapImpl {
public:
    MMapImplMio(const std::string &filename) : impl(filename, 0, mio::map_entire_file) {}
    virtual size_t size() override {
        return impl.size();
    }
    virtual const char *data() override {
        return impl.data();
    }

private:
    mio::mmap_source impl;
};

#ifdef __linux__ 

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

class SafeTensors::MMapImplPrivate : public SafeTensors::MMapImpl {
public:
    MMapImplPrivate(const std::string &filename) {
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), filename);
        }

        struct stat statbuf;
        fstat(fd, &statbuf);
        filesize = statbuf.st_size;

        ptr = mmap(0, filesize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            throw std::system_error(errno, std::generic_category(), filename);
        }

        close(fd);
    }
    ~MMapImplPrivate() {
        munmap(ptr, filesize);
    }

    virtual size_t size() override {
        return filesize;
    }
    virtual const char *data() override {
        return (const char *)ptr;
    }

private:
    size_t filesize;
    void *ptr;
};

#else 

class SafeTensors::MMapImplPrivate : public SafeTensors::MMapImpl {
public:
    MMapImplPrivate(const std::string &filename) {
        throw std::runtime_error("MAP_PRIVATE is not implemented on this system");
    }

    virtual size_t size() override {
        return 0;
    }
    virtual const char *data() override {
        return nullptr;
    }
};

#endif

SafeTensors::SafeTensors(const std::string &filename) {
    this->mapped = std::make_unique<MMapImplMio>(filename);

    if (cudaHostRegister(const_cast<char *>(this->mapped->data()), this->mapped->size(), cudaHostRegisterPortable | cudaHostRegisterReadOnly) != cudaSuccess) {
        spdlog::warn("Unable to pin memory: {}", cudaGetErrorString(cudaGetLastError()));
        // mlock(const_cast<char *>(this->mapped->data()), this->mapped->size());
#ifdef __linux__
        spdlog::info("Try MAP_PRIVATE");
        this->mapped.reset();
        this->mapped = std::make_unique<MMapImplPrivate>(filename);
        checkCUDA(cudaHostRegister(const_cast<char *>(this->mapped->data()), this->mapped->size(), cudaHostRegisterPortable));
#endif
    }

    parseHeader();
}

SafeTensors::~SafeTensors() {
    checkCUDA(cudaHostUnregister(const_cast<char *>(this->mapped->data())));
}

void SafeTensors::parseHeader() {
    static const std::unordered_map<std::string, Tensor::ScalarType> mapDType = {
        { "BF16", Tensor::BF16  },
        { "F16",  Tensor::FP16  },
        { "F32",  Tensor::FP32  },
        { "I8",   Tensor::INT8  },
        { "I32",  Tensor::INT32 },
        { "I64",  Tensor::INT64 },
    };

    auto check = [](bool cond, std::source_location location = std::source_location::current()) {
        if (!cond) {
            throw std::runtime_error(format("Safetensors check failed at {}:{}", location.file_name(), location.line()));
        }
    };

    check(this->mapped->size() > 8);
    uint64_t sizeHeader = *reinterpret_cast<const uint64_t *>(this->mapped->data());

    check(this->mapped->size() - 8 >= sizeHeader);
    json header = json::parse(this->mapped->data() + 8, this->mapped->data() + 8 + sizeHeader);

    const uint64_t offsetMax = this->mapped->size() - sizeHeader - 8;
    std::set<size_t> offsets;

    for (auto &&[key, info] : header.items()) {
        if (key == "__metadata__") {
            continue;
        }

        auto dtype = mapDType.at(info["dtype"].get<std::string>());;
        auto shape = info["shape"].get<std::vector<int>>();
        auto data_offsets = info["data_offsets"].get<std::vector<uint64_t>>();

        check(data_offsets.size() == 2);
        check(data_offsets[0] <= data_offsets[1]);
        check(data_offsets[0] < offsetMax);
        check(data_offsets[1] <= offsetMax);
        for (auto &&dim : shape) {
            check(dim >= 0);
        }

        TensorInfo tinfo;
        tinfo.type = dtype;
        tinfo.shape = TensorShape(shape);
        tinfo.length = data_offsets[1] - data_offsets[0];
        tinfo.offset = 8 + sizeHeader + data_offsets[0];

        // TODO: check range overlap
        check(!offsets.contains(tinfo.offset));
        offsets.insert(tinfo.offset);

        check(tinfo.shape.size() * Tensor::scalarSize.at(tinfo.type) <= tinfo.length);

        tensors[key] = tinfo;
    }
}

Tensor SafeTensors::getTensor(const std::string &key) {
    if (!tensors.contains(key)) {
        return Tensor{};
    }
    TensorInfo &info = tensors.at(key);

    std::shared_ptr<BufferMMap> buffer = info.buffer.lock();
    if (!buffer) {
        buffer = std::make_shared<BufferMMap>(const_cast<char *>(this->mapped->data() + info.offset), info.length, shared_from_this());
        info.buffer = buffer;
    }

    Tensor result;
    result.shape = info.shape;
    result.scalarType = info.type;
    result.buffer = buffer;

    return result;
}

