import os

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    fp = open("nunchaku/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    ROOT_DIR = os.path.dirname(__file__)

    INCLUDE_DIRS = [
        "src",
        "third_party/cutlass/include",
        "third_party/json/include",
        "third_party/mio/include",
        "third_party/spdlog/include",
    ]

    INCLUDE_DIRS = ["-I" + ROOT_DIR + "/" + dir for dir in INCLUDE_DIRS]

    DEBUG = True

    def ncond(s) -> list:
        if DEBUG:
            return []
        else:
            return [s]

    def cond(s) -> list:
        if DEBUG:
            return [s]
        else:
            return []

    CXX_FLAGS = [
        "-DBUILD_NUNCHAKU=1",
        "-fvisibility=hidden",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Og",
        *INCLUDE_DIRS,
    ]
    NVCC_FLAGS = [
        "-DBUILD_NUNCHAKU=1",
        "-gencode",
        "arch=compute_86,code=sm_86",
        "-gencode",
        "arch=compute_89,code=sm_89",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Xcudafe",
        "--diag_suppress=20208",  # spdlog: 'long double' is treated as 'double' in device code
        *cond("-G"),
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--threads=2",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--generate-line-info",
        "--ptxas-options=--allow-expensive-optimizations=true",
        *INCLUDE_DIRS,
    ]

    nunchaku_extension = CUDAExtension(
        name="nunchaku._C",
        sources=[
            "nunchaku/csrc/pybind.cpp",
            "src/interop/torch.cpp",
            "src/activation.cpp",
            "src/layernorm.cpp",
            "src/Linear.cpp",
            *ncond("src/FluxModel.cpp"),
            "src/Serialization.cpp",
            *ncond("src/kernels/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_block_hdim64_fp16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_block_hdim64_bf16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_block_hdim128_fp16_sm80.cu"),
            *ncond("src/kernels/flash_attn/src/flash_fwd_block_hdim128_bf16_sm80.cu"),
            "src/kernels/activation_kernels.cu",
            "src/kernels/layernorm_kernels.cu",
            "src/kernels/misc_kernels.cu",
            "src/kernels/gemm_w4a4.cu",
            "src/kernels/gemm_batched.cu",
            "src/kernels/gemm_f16.cu",
            "src/kernels/awq/gemv_awq.cu",
            *ncond("src/kernels/flash_attn/flash_api.cpp"),
            *ncond("src/kernels/flash_attn/flash_api_adapter.cpp"),
        ],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )

    setuptools.setup(
        name="nunchaku",
        version=version,
        packages=setuptools.find_packages(),
        ext_modules=[nunchaku_extension],
        cmdclass={"build_ext": BuildExtension},
    )
