#pragma once

#include "common.h"
#include "Tensor.h"

Tensor
gemv_awq(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, int m, int n, int k, int group_size);
