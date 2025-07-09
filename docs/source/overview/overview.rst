Project Structure Overview
==========================

**Nunchaku** is a high-performance inference engine optimized for 4-bit quantized neural networks, designed to accelerate diffusion models and LLMs.

Architecture Overview
---------------------

Nunchaku combines:

- **C++ Backend**: High-performance CUDA kernels and core inference engine
- **Python Frontend**: User-friendly API compatible with popular ML frameworks
- **Quantization Support**: Specialized handling for 4-bit quantized models
- **Multi-Model Support**: Optimized implementations for FLUX.1 and SANA models

Directory Structure
-------------------

**Core Components**

.. code-block:: text

    nunchaku/
    ├── src/                       # C++ implementation
    │   ├── FluxModel.cpp/h        # FLUX.1 model
    │   ├── SanaModel.cpp/h        # SANA model
    │   ├── Linear.cpp/h           # Linear layers
    │   ├── Tensor.h               # Tensor operations
    │   ├── kernels/               # CUDA kernels
    │   └── interop/               # Interoperability
    ├── nunchaku/                  # Python package
    │   ├── models/                # Model implementations
    │   ├── pipeline/              # Inference pipelines
    │   ├── lora/                  # LoRA support
    │   └── caching/               # Caching mechanisms
    ├── examples/                  # Usage examples
    ├── tests/                     # Test suite
    ├── docs/                      # Documentation
    └── third_party/               # External dependencies

**Key Directories**

- ``src/``: C++ backend with CUDA kernels for quantized operations
- ``nunchaku/``: Python API and model implementations
- ``examples/``: Ready-to-use examples for different models and features
- ``tests/``: Comprehensive test suite
- ``docs/``: Documentation and guides

Key Features
------------

**Model Support**
- FLUX.1 (dev, schnell, and variants)
- SANA (1.6B and related models)
- 4-bit weight quantization
- LoRA fine-tuning

**Performance**
- Custom CUDA kernels for quantized operations
- Efficient memory management and caching
- Optimized tensor operations
- Block sparse attention

**Integration**
- Hugging Face Diffusers compatibility
- ControlNet support
- ComfyUI integration
- PuLID support

Getting Started
---------------

1. **Examples**: Start with ``examples/`` for basic usage patterns
2. **Models**: Explore ``nunchaku/models/`` for implementations
3. **Documentation**: Read ``docs/source/`` for detailed guides
4. **Tests**: Check ``tests/`` for validation examples

Build Process
-------------

The build involves:
1. C++ compilation with CUDA kernels
2. Python bindings via pybind11
3. Wheel generation for distribution
4. Comprehensive testing

This structure enables efficient development and deployment of high-performance quantized neural network inference while maintaining compatibility with existing ML frameworks. 