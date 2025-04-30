#pragma once

#include "common.h"
#include "Tensor.h"

class DebugContext {
public:
    DebugContext() {
        ctxs.insert(this);
    }
    DebugContext(const DebugContext &) = delete;
    DebugContext(DebugContext &&)      = delete;

    ~DebugContext() {
        ctxs.erase(this);
    }

    std::map<std::string, Tensor> tensors;

    static inline thread_local std::set<DebugContext *> ctxs;
};
