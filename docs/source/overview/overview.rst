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

**Main Components**

The project consists of three key folders:

**``nunchaku/`` - Main Python Package**
    This is the primary Python package that users interact with. It contains:
    
    - Model implementations for FLUX.1 and SANA
    - Inference pipelines for different use cases
    - LoRA (Low-Rank Adaptation) support for fine-tuning
    - Caching mechanisms for improved performance
    - Utility functions and helper modules
    - The compiled C++ extension (``_C.*.so``)

**``src/`` - C++ Backend Library**
    Houses the high-performance C++ implementation that powers the Python package:
    
    - FLUX.1 and SANA model implementations in C++
    - Linear layer operations and tensor utilities
    - Custom CUDA kernels for quantized operations
    - Memory management and optimization routines
    - Python-C++ interoperability layer
    - Serialization and model loading functionality

**``app/`` - Gradio Demo Applications**
    Interactive web-based demos built with Gradio:
    
    - FLUX.1 demos for different model variants
    - SANA model demonstrations
    - User-friendly interfaces for trying out features
    - Real-time inference examples

**Supporting Directories**

- **``examples/``** - Usage examples and tutorials showing how to use different features
- **``tests/``** - Comprehensive test suite for validation and quality assurance
- **``docs/``** - Documentation source files and build configuration
- **``third_party/``** - External dependencies and submodules

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
2. **Demos**: Try ``app/`` for interactive Gradio demos
3. **Models**: Explore ``nunchaku/models/`` for implementations
4. **Documentation**: Read ``docs/source/`` for detailed guides
5. **Tests**: Check ``tests/`` for validation examples

Build Process
-------------

The build involves:
1. C++ compilation with CUDA kernels (``src/`` → ``nunchaku/_C.*.so``)
2. Python bindings via pybind11
3. Wheel generation for distribution
4. Comprehensive testing

This structure enables efficient development and deployment of high-performance quantized neural network inference while maintaining compatibility with existing ML frameworks. 