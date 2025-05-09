import os
import re
import subprocess
import sys

import setuptools
import torch
from packaging import version as packaging_version
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            if not "cxx" in ext.extra_compile_args:
                ext.extra_compile_args["cxx"] = []
            if not "nvcc" in ext.extra_compile_args:
                ext.extra_compile_args["nvcc"] = []
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["msvc"]
                ext.extra_compile_args["nvcc"] += ext.extra_compile_args["nvcc_msvc"]
            else:
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["gcc"]
        super().build_extensions()


def get_cuda_home() -> str:
    """Get CUDA home directory with better diagnostics."""
    # Common CUDA installation paths
    common_paths = [
        "/usr/local/cuda",
        "/usr/cuda",
        "/opt/cuda",
    ]
    
    # First try environment variables
    for env_var in ["CUDA_HOME", "CUDA_PATH"]:
        cuda_home = os.getenv(env_var)
        if cuda_home and os.path.exists(os.path.join(cuda_home, "bin/nvcc")):
            print(f"Found CUDA via {env_var}: {cuda_home}")
            return cuda_home

    # Then try PyTorch's CUDA_HOME
    if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "bin/nvcc")):
        print(f"Found CUDA via PyTorch: {CUDA_HOME}")
        return CUDA_HOME

    # Finally try common paths
    for path in common_paths:
        if os.path.exists(os.path.join(path, "bin/nvcc")):
            print(f"Found CUDA in common path: {path}")
            return path

    # If we get here, try to find any nvcc binary
    for path in common_paths:
        if os.path.exists(path):
            print(f"CUDA directory exists but nvcc not found: {path}")
            print(f"Contents of {os.path.join(path, 'bin')}:")
            try:
                print(os.listdir(os.path.join(path, "bin")))
            except:
                print("(cannot list directory)")

    raise Exception(
        "CUDA installation not found. Please ensure CUDA is installed and one of these is set:\n"
        "- CUDA_HOME environment variable\n"
        "- CUDA_PATH environment variable\n"
        "Or CUDA is installed in one of: " + ", ".join(common_paths)
    )


def get_sm_targets() -> list[str]:
    """Get list of target SM architectures."""
    cuda_home = get_cuda_home()
    nvcc_path = os.path.join(cuda_home, "bin/nvcc")
    
    try:
        nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
        print(f"nvcc found at: {nvcc_path}")
        print(f"nvcc output:\n{nvcc_output}")
        
        match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", nvcc_output)
        if match:
            nvcc_version = match.group(2)
            print(f"Parsed nvcc version: {nvcc_version}")
        else:
            raise Exception("Could not parse nvcc version from output: " + nvcc_output)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running {nvcc_path}: {e}")
    except Exception as e:
        raise Exception(f"Error getting nvcc version: {e}")

    support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")

    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":
        ret = []
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            sm = f"{capability[0]}{capability[1]}"
            if sm == "120" and support_sm120:
                sm = "120a"
            assert sm in ["75", "80", "86", "89", "120a"], f"Unsupported SM {sm}"
            if sm not in ret:
                ret.append(sm)
    else:
        assert install_mode == "ALL"
        ret = ["75", "80", "86", "89"]
        if support_sm120:
            ret.append("120a")
    return ret


if __name__ == "__main__":
    fp = open("nunchaku/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    torch_version = torch.__version__.split("+")[0]
    torch_major_minor_version = ".".join(torch_version.split(".")[:2])
    version = version + "+torch" + torch_major_minor_version

    ROOT_DIR = os.path.dirname(__file__)

    INCLUDE_DIRS = [
        "src",
        "third_party/cutlass/include",
        "third_party/json/include",
        "third_party/mio/include",
        "third_party/spdlog/include",
        "third_party/Block-Sparse-Attention/csrc/block_sparse_attn",
    ]

    INCLUDE_DIRS = [os.path.join(ROOT_DIR, dir) for dir in INCLUDE_DIRS]

    DEBUG = False

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

    sm_targets = get_sm_targets()
    print(f"Detected SM targets: {sm_targets}", file=sys.stderr)

    assert len(sm_targets) > 0, "No SM targets found"

    GCC_FLAGS = ["-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-fvisibility=hidden", "-g", "-std=c++20", "-UNDEBUG", "-Og"]
    MSVC_FLAGS = ["/DENABLE_BF16=1", "/DBUILD_NUNCHAKU=1", "/std:c++20", "/UNDEBUG", "/Zc:__cplusplus", "/FS"]
    NVCC_FLAGS = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
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
        f"--threads={len(sm_targets)}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=--allow-expensive-optimizations=true",
    ]

    if os.getenv("NUNCHAKU_BUILD_WHEELS", "0") == "0":
        NVCC_FLAGS.append("--generate-line-info")

    for target in sm_targets:
        NVCC_FLAGS += ["-gencode", f"arch=compute_{target},code=sm_{target}"]

    NVCC_MSVC_FLAGS = ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS", "-Xcompiler", "/bigobj"]

    nunchaku_extension = CUDAExtension(
        name="nunchaku._C",
        sources=[
            "nunchaku/csrc/pybind.cpp",
            "src/interop/torch.cpp",
            "src/activation.cpp",
            "src/layernorm.cpp",
            "src/Linear.cpp",
            *ncond("src/FluxModel.cpp"),
            *ncond("src/OminiFluxModel.cpp"),
            *ncond("src/SanaModel.cpp"),
            "src/Serialization.cpp",
            "src/Module.cpp",
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_bf16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_bf16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_bf16_sm80.cu"),
            *ncond(
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_fp16_sm80.cu"
            ),
            *ncond(
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_bf16_sm80.cu"
            ),
            "src/kernels/activation_kernels.cu",
            "src/kernels/layernorm_kernels.cu",
            "src/kernels/misc_kernels.cu",
            "src/kernels/zgemm/gemm_w4a4.cu",
            "src/kernels/zgemm/gemm_w4a4_test.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_fp16_fp4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_bf16_fp4.cu",
            "src/kernels/zgemm/gemm_w8a8.cu",
            "src/kernels/zgemm/attention.cu",
            "src/kernels/dwconv.cu",
            "src/kernels/gemm_batched.cu",
            "src/kernels/gemm_f16.cu",
            "src/kernels/awq/gemm_awq.cu",
            "src/kernels/awq/gemv_awq.cu",
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api.cpp"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api_adapter.cpp"),
        ],
        extra_compile_args={"gcc": GCC_FLAGS, "msvc": MSVC_FLAGS, "nvcc": NVCC_FLAGS, "nvcc_msvc": NVCC_MSVC_FLAGS},
        include_dirs=INCLUDE_DIRS,
    )

    setuptools.setup(
        name="nunchaku",
        version=version,
        packages=setuptools.find_packages(),
        ext_modules=[nunchaku_extension],
        cmdclass={"build_ext": CustomBuildExtension},
    )
