#pragma once

#include "common.h"
#include "Tensor.h"

Tensor gemm_w8a8_fp16(Tensor input,  // INT8
                      Tensor weight, // INT8
                      Tensor out,
                      half scale,
                      half bias);
