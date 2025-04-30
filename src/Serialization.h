#pragma once

#include "common.h"
#include "Tensor.h"

class BufferMMap : public Buffer {
public:
    BufferMMap(void *ptr, size_t size, std::shared_ptr<void> parent) : parent(parent) {
        this->size        = size;
        this->device.type = Device::CPU;
        this->ptr         = ptr;
        // auto ret = cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
        // if (ret == cudaSuccess) {
        //     this->registered = true;
        // } else {
        //     log(std::format("cudaHostRegister failed at {:p} (size={}): {}", ptr, size,
        //     cudaGetErrorString(cudaGetLastError()))); this->registered = false;
        // }
    }
    virtual ~BufferMMap() {
        // if (registered) {
        //     checkCUDA(cudaHostUnregister(ptr));
        // }
    }

public:
    std::shared_ptr<void> parent;
    // bool registered;
};

class SafeTensors : public TensorsProvider, public std::enable_shared_from_this<SafeTensors> {
public:
    SafeTensors(const std::string &filename);
    ~SafeTensors();

    virtual bool contains(const std::string &key) const override {
        return tensors.contains(key);
    }
    virtual Tensor getTensor(const std::string &key) override;

private:
    void parseHeader();

private:
    class MMapImpl;
    class MMapImplMio;
    class MMapImplPrivate;
    class MMapImplRead;

    struct TensorInfo {
        TensorShape shape;
        Tensor::ScalarType type;
        size_t offset;
        size_t length;
        std::weak_ptr<BufferMMap> buffer;
    };
    std::map<std::string, TensorInfo> tensors;
    std::unique_ptr<MMapImpl> mapped;

    bool hostRegistered, memoryPinned;
};
