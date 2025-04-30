#pragma once

#include "common.h"
#include "Tensor.h"
#include "debug.h"

class Module {
protected:
    enum class ParamFlags : int {
        None     = 0,
        Optional = 1,
        LazyLoad = 2,
    };
    struct TensorLazyLoadInfo {
        TensorShape shape;
        Tensor::ScalarType type;
        Device device;

        Tensor src;
    };
    struct Param {
        Tensor *tensor   = nullptr;
        ParamFlags flags = ParamFlags::None;

        TensorLazyLoadInfo lazyInfo;
    };

    friend inline ParamFlags operator|(ParamFlags lhs, ParamFlags rhs) {
        return static_cast<ParamFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
    }
    friend inline ParamFlags operator&(ParamFlags lhs, ParamFlags rhs) {
        return static_cast<ParamFlags>(static_cast<int>(lhs) & static_cast<int>(rhs));
    }
    static bool checkFlag(ParamFlags flags, ParamFlags target) {
        return int(flags & target);
    }

public:
    std::string getFullName() const {
        if (!parent) {
            return name;
        }
        std::string fullName = parent->getFullName();
        if (fullName.empty()) {
            return name;
        } else {
            return fullName + "." + name;
        }
    }

    std::string getPrefix() const {
        std::string fullName = getFullName();
        std::string prefix   = fullName.empty() ? "" : fullName + ".";
        return prefix;
    }

    void traverse(std::function<void(Module *)> func) {
        func(this);
        for (Module *c : this->children) {
            c->traverse(func);
        }
    }

    virtual void loadParams(TensorsProvider &provider, bool partial = false) {
        for (Module *c : children) {
            c->loadParams(provider, partial);
        }
        std::string prefix = getPrefix();
        for (auto &&[key, param] : params) {
            Tensor src = provider.getTensor(prefix + key);
            if (!src.valid()) {
                if (partial || int(param.flags & ParamFlags::Optional)) {
                    continue;
                }
                throw std::runtime_error(spdlog::fmt_lib::format("Tensor {} not found", prefix + key));
            }
            if (enabledLazyLoad && checkFlag(param.flags, ParamFlags::LazyLoad)) {
                param.lazyInfo.src = src;
                if (!param.tensor->valid()) {
                    continue;
                }
                // keep loading params if param is not released
            }
            this->loadParam(key, *param.tensor, src);
            // tensor->copy_(src);
        }
    }

    void setName(std::string name) {
        assert(!parent);
        this->name = std::move(name);
    }

    void loadLazyParams() {
        traverse([](Module *m) {
            for (auto &&[key, param] : m->params) {
                if (!checkFlag(param.flags, ParamFlags::LazyLoad)) {
                    continue;
                }

                TensorLazyLoadInfo &lazy = param.lazyInfo;
                Tensor &dst              = *param.tensor;
                Tensor src               = lazy.src;

                if (dst.valid()) {
                    continue;
                }
                dst = Tensor::allocate(lazy.shape, lazy.type, lazy.device);

                if (!src.valid() && !checkFlag(param.flags, ParamFlags::Optional)) {
                    throw std::runtime_error(
                        spdlog::fmt_lib::format("Lazy load: Tensor {} has no src", m->getPrefix() + key));
                }
                m->loadParam(key, dst, src);
            }
        });
    }
    void releaseLazyParams() {
        traverse([](Module *m) {
            if (!m->enabledLazyLoad) {
                return;
            }
            for (auto &&[key, param] : m->params) {
                if (checkFlag(param.flags, ParamFlags::LazyLoad)) {
                    *param.tensor = Tensor{};
                }
            }
        });
    }
    void setLazyLoad(bool val) {
        traverse([val](Module *m) { m->enabledLazyLoad = val; });
    }
    void setAutoCastFP16(bool val) {
        traverse([val](Module *m) { m->enabledAutoCastFP16 = val; });
    }

protected:
    virtual void loadParam(std::string key, Tensor &dst, Tensor src) {
        static const std::set<Tensor::ScalarType> whitelist = {
            Tensor::FP16,
            Tensor::BF16,
        };
        if (enabledAutoCastFP16 && dst.scalar_type() != src.scalar_type() && whitelist.contains(dst.scalar_type()) &&
            whitelist.contains(src.scalar_type())) {
            copyWithCast(dst, src);
        } else {
            dst.copy_(src);
        }
    }

    struct ChildrenRegisterHelper {
        ChildrenRegisterHelper(Module &self) : self(self) {}
        Module &self;
        ChildrenRegisterHelper operator()(Module &module, std::string name) {
            return self.registerChildren(module, name);
        }
    };
    ChildrenRegisterHelper registerChildren(Module &module, std::string name) {
        module.parent = this;
        module.name   = name;
        children.push_back(&module);
        return ChildrenRegisterHelper(*this);
    }

    struct ParamsRegisterHelper {
        ParamsRegisterHelper(Module &self) : self(self) {}
        Module &self;
        ParamsRegisterHelper operator()(Tensor &param, std::string name, ParamFlags flags = ParamFlags::None) {
            return self.registerParams(param, name, flags);
        }
    };
    ParamsRegisterHelper registerParams(Tensor &param, std::string name, ParamFlags flags = ParamFlags::None) {
        if (param.valid()) {
            params[name].tensor = &param;
            params[name].flags  = flags;

            if (checkFlag(flags, ParamFlags::LazyLoad) && param.valid()) {
                TensorLazyLoadInfo &lazy = params[name].lazyInfo;
                lazy.shape               = param.shape;
                lazy.type                = param.dtype();
                lazy.device              = param.device();
            }
        }
        return ParamsRegisterHelper(*this);
    }

    void debug(std::string name, Tensor tensor) {
        if (DebugContext::ctxs.empty() || !tensor.valid()) {
            return;
        }
        std::string prefix = getFullName();
        if (!prefix.empty()) {
            prefix += ".";
        }
        tensor = tensor.copy(Device::cpu());
        for (auto &&ctx : DebugContext::ctxs) {
            ctx->tensors[prefix + name] = tensor;
        }
    }

private:
    void copyWithCast(Tensor dst, Tensor src);

public:
    Module *parent   = nullptr;
    std::string name = "";
    std::vector<Module *> children;
    std::map<std::string, Param> params;

    bool enabledLazyLoad     = false;
    bool enabledAutoCastFP16 = true;
};

struct LayerOffloadHelper {
    using func_t = std::function<void(int)>;

    const bool offload;
    const int numLayers;

    func_t funcCompute, funcLoad, funcUnload;

    std::unique_ptr<CUDAStreamWrapper> streamCompute;
    std::unique_ptr<CUDAStreamWrapper> streamLoad;
    std::unique_ptr<CUDAEventWrapper> eventComputeDone;
    std::unique_ptr<CUDAEventWrapper> eventLoadDone;

    LayerOffloadHelper(bool offload, int numLayers, func_t funcCompute, func_t funcLoad, func_t funcUnload)
        : offload(offload), numLayers(numLayers), funcCompute(funcCompute), funcLoad(funcLoad), funcUnload(funcUnload) {
        if (offload) {
            streamCompute = std::make_unique<CUDAStreamWrapper>();
            streamLoad    = std::make_unique<CUDAStreamWrapper>();

            needWorkaround = checkWorkaround();
            if (needWorkaround) {
                spdlog::debug("Offloading helper: use WDDM workaround");
            }
        }
    }

    void run() {
        for (int i = 0; i < numLayers; i++) {
            run(i);
        }
        waitEvent(eventComputeDone.get());
        funcUnload(numLayers - 1);
    }

private:
    void run(int layer) {
        if (!offload) {
            funcCompute(layer);
        } else {
            std::unique_ptr<CUDAEventWrapper> nextComputeDone, nextLoadDone;

            // issue compute kernels first so that we could still overlap compute and memcpy if memory is not pinned
            {
                CUDAStreamContext ctx(streamCompute->stream);
                waitEvent(eventLoadDone.get());
                funcCompute(layer);
                nextComputeDone = std::make_unique<CUDAEventWrapper>();
                checkCUDA(cudaEventRecord(nextComputeDone->event, getCurrentCUDAStream()));
                workaroundFlush();
            }

            {
                CUDAStreamContext ctx(streamLoad->stream);
                waitEvent(eventComputeDone.get());
                if (layer - 1 > 0) {
                    funcUnload(layer - 1);
                }
                if (layer + 1 < numLayers) {
                    funcLoad(layer + 1);
                }
                nextLoadDone = std::make_unique<CUDAEventWrapper>();
                checkCUDA(cudaEventRecord(nextLoadDone->event, getCurrentCUDAStream()));
                workaroundFlush();
            }

            eventComputeDone = std::move(nextComputeDone);
            eventLoadDone    = std::move(nextLoadDone);

            workaroundSynchronize();
        }
    }

    static void waitEvent(CUDAEventWrapper *event) {
        if (!event) {
            return;
        }
        checkCUDA(cudaStreamWaitEvent(getCurrentCUDAStream(), event->event));
    }

    // WDDM prevents multiple streams run concurrently
    // use flush and synchronize to work around
    bool needWorkaround;
    static bool checkWorkaround() {
        if (char *env = getenv("NUNCHAKU_OFFLOAD_WDDM_WORKAROUND")) {
            if (std::string(env) == "1") {
                return true;
            } else if (std::string(env) == "0") {
                return false;
            }
        }

#ifdef _WIN32
        return true;
#else
        return false;
#endif
    }
    void workaroundFlush() {
        if (!needWorkaround) {
            return;
        }
        cudaStreamQuery(getCurrentCUDAStream());
    }
    void workaroundSynchronize() {
        if (!needWorkaround) {
            return;
        }
        checkCUDA(cudaEventSynchronize(eventComputeDone->event));
    }
};
