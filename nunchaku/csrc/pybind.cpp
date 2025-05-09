#include "gemm.h"
#include "gemm88.h"
#include "flux.h"
#include "sana.h"
#include "ops.h"
#include "utils.h"
#include <torch/extension.h>
#include "interop/torch.h"
#include <pybind11/pybind11.h>
#include "OminiFluxModel.h"

// This PYBIND11_MODULE block defines the Python bindings for the C++ Nunchaku library.
// It exposes C++ classes and functions to Python, allowing them to be used as if they were native Python objects.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bindings for the QuantizedFluxModel class (likely an earlier or different version of Flux model)
    py::class_<QuantizedFluxModel>(m, "QuantizedFluxModel")
        .def(py::init<>()) // Exposes the default constructor
        // Exposes the 'init' method for model initialization
        .def("init",
             &QuantizedFluxModel::init,
             py::arg("use_fp4"),      // Argument: whether to use FP4 quantization
             py::arg("offload"),      // Argument: whether to enable layer offloading
             py::arg("bf16"),         // Argument: whether to use bfloat16 precision
             py::arg("deviceId"))   // Argument: CUDA device ID
        // Exposes 'set_residual_callback' to allow Python functions to be called from C++
        .def("set_residual_callback",
             [](QuantizedFluxModel &self, pybind11::object call_back) {
                 if (call_back.is_none()) {
                     self.set_residual_callback(pybind11::function()); // Clear callback if None
                 } else {
                     self.set_residual_callback(call_back); // Set the Python callback
                 }
             })
        .def("reset", &QuantizedFluxModel::reset) // Exposes model reset functionality
        .def("load", &QuantizedFluxModel::load, py::arg("path"), py::arg("partial") = false) // Load model from path
        .def("loadDict", &QuantizedFluxModel::loadDict, py::arg("dict"), py::arg("partial") = false) // Load model from a Python dictionary (state_dict)
        // Exposes the main 'forward' pass of the model
        .def("forward",
             &QuantizedFluxModel::forward,
             py::arg("hidden_states"),
             py::arg("encoder_hidden_states"),
             py::arg("temb"),
             py::arg("rotary_emb_img"),
             py::arg("rotary_emb_context"),
             py::arg("rotary_emb_single"),
             py::arg("controlnet_block_samples")        = py::none(), // Optional ControlNet features
             py::arg("controlnet_single_block_samples") = py::none(), // Optional ControlNet features for single blocks
             py::arg("skip_first_layer")                = false)     // Optional: skip the first layer
        // Exposes 'forward_layer' for running a single layer
        .def("forward_layer",
             &QuantizedFluxModel::forward_layer,
             py::arg("idx"), // Layer index
             py::arg("hidden_states"),
             py::arg("encoder_hidden_states"),
             py::arg("temb"),
             py::arg("rotary_emb_img"),
             py::arg("rotary_emb_context"),
             py::arg("controlnet_block_samples")        = py::none(),
             py::arg("controlnet_single_block_samples") = py::none())
        .def("forward_single_layer", &QuantizedFluxModel::forward_single_layer) // Forward pass for a single type of block
        .def("norm_one_forward", &QuantizedFluxModel::norm_one_forward) // Forward pass for the first norm layer of a block
        .def("startDebug", &QuantizedFluxModel::startDebug) // Debugging utilities
        .def("stopDebug", &QuantizedFluxModel::stopDebug)
        .def("getDebugResults", &QuantizedFluxModel::getDebugResults)
        .def("setLoraScale", &QuantizedFluxModel::setLoraScale) // Set LoRA scaling factor
        .def("setAttentionImpl", &QuantizedFluxModel::setAttentionImpl) // Set attention implementation (e.g., flashattn2)
        .def("isBF16", &QuantizedFluxModel::isBF16); // Check if model is using bfloat16
    py::class_<QuantizedSanaModel>(m, "QuantizedSanaModel")
        .def(py::init<>())
        .def("init",
             &QuantizedSanaModel::init,
             py::arg("config"),
             py::arg("pag_layers"),
             py::arg("use_fp4"),
             py::arg("bf16"),
             py::arg("deviceId"))
        .def("reset", &QuantizedSanaModel::reset)
        .def("load", &QuantizedSanaModel::load, py::arg("path"), py::arg("partial") = false)
        .def("loadDict", &QuantizedSanaModel::loadDict, py::arg("dict"), py::arg("partial") = false)
        .def("forward", &QuantizedSanaModel::forward)
        .def("forward_layer", &QuantizedSanaModel::forward_layer)
        .def("startDebug", &QuantizedSanaModel::startDebug)
        .def("stopDebug", &QuantizedSanaModel::stopDebug)
        .def("getDebugResults", &QuantizedSanaModel::getDebugResults);
    py::class_<QuantizedGEMM>(m, "QuantizedGEMM")
        .def(py::init<>())
        .def("init", &QuantizedGEMM::init)
        .def("reset", &QuantizedGEMM::reset)
        .def("load", &QuantizedGEMM::load)
        .def("forward", &QuantizedGEMM::forward)
        .def("quantize", &QuantizedGEMM::quantize)
        .def("startDebug", &QuantizedGEMM::startDebug)
        .def("stopDebug", &QuantizedGEMM::stopDebug)
        .def("getDebugResults", &QuantizedGEMM::getDebugResults);
    py::class_<Tensor>(m, "Tensor");
    py::class_<QuantizedGEMM88>(m, "QuantizedGEMM88")
        .def(py::init<>())
        .def("init", &QuantizedGEMM88::init)
        .def("reset", &QuantizedGEMM88::reset)
        .def("load", &QuantizedGEMM88::load)
        .def("forward", &QuantizedGEMM88::forward)
        .def("startDebug", &QuantizedGEMM88::startDebug)
        .def("stopDebug", &QuantizedGEMM88::stopDebug)
        .def("getDebugResults", &QuantizedGEMM88::getDebugResults);
    py::class_<QuantizedOminiFluxModel>(m, "QuantizedOminiFluxModel")
        .def(py::init<>())
        .def("init",
             &QuantizedOminiFluxModel::init,
             py::arg("use_fp4"),      // Use 4-bit precision for quantization
             py::arg("offload"),      // Enable layer offloading to CPU
             py::arg("bf16"),         // Use bfloat16, otherwise float16
             py::arg("deviceId"))   // CUDA device ID
        .def("set_residual_callback",
             [](QuantizedOminiFluxModel &self, pybind11::object call_back) {
                 if (call_back.is_none()) {
                     self.set_residual_callback(pybind11::function()); // Clear callback if None
                 } else {
                     self.set_residual_callback(call_back); // Set the Python callback
                 }
             })
        .def("reset", &QuantizedOminiFluxModel::reset) // Resets the model state
        .def("load", &QuantizedOminiFluxModel::load, py::arg("path"), py::arg("partial") = false) // Load weights from a file path
        .def("loadDict", &QuantizedOminiFluxModel::loadDict, py::arg("dict"), py::arg("partial") = false) // Load weights from a Python dictionary (state_dict like)
        // Defines the main forward pass with all its arguments.
        .def("forward",
             &QuantizedOminiFluxModel::forward,
             py::arg("hidden_states"),         // Image latents
             py::arg("cond_hidden_states"),    // Conditional latents
             py::arg("encoder_hidden_states"), // Text encoder latents
             py::arg("temb"),                  // Time embeddings
             py::arg("cond_temb"),             // Conditional time embeddings
             py::arg("rotary_emb_img"),        // Rotary embeddings for image
             py::arg("rotary_emb_context"),    // Rotary embeddings for text/context
             py::arg("rotary_emb_single"),     // Rotary embeddings for single/concatenated sequence (later stages)
             py::arg("rotary_emb_cond"),       // Rotary embeddings for conditional inputs
             py::arg("controlnet_block_samples")        = py::none(), // Optional ControlNet features for joint blocks
             py::arg("controlnet_single_block_samples") = py::none(), // Optional ControlNet features for single blocks
             py::arg("skip_first_layer")                = false)     // Option to skip the first layer
        // Defines the forward pass for a single specified layer.
        .def("forward_layer",
             &QuantizedOminiFluxModel::forward_layer,
             py::arg("idx"),                   // Index of the layer to run
             py::arg("hidden_states"),
             py::arg("cond_hidden_states"),
             py::arg("encoder_hidden_states"),
             py::arg("temb"),
             py::arg("cond_temb"),
             py::arg("rotary_emb_img"),
             py::arg("rotary_emb_context"),
             py::arg("rotary_emb_cond"),
             py::arg("controlnet_block_samples")        = py::none(),
             py::arg("controlnet_single_block_samples") = py::none())
        // Exposes the forward method of the first AdaLayerNorm in a specified block (used for caching like TeaCache).
        .def("norm_one_forward", &QuantizedOminiFluxModel::norm_one_forward)
        .def("startDebug", &QuantizedOminiFluxModel::startDebug)     // Debug utilities
        .def("stopDebug", &QuantizedOminiFluxModel::stopDebug)
        .def("getDebugResults", &QuantizedOminiFluxModel::getDebugResults)
        .def("setLoraScale", &QuantizedOminiFluxModel::setLoraScale) // Sets the scaling factor for LoRA layers
        .def("setAttentionImpl", &QuantizedOminiFluxModel::setAttentionImpl) // Chooses attention kernel (e.g., "flashattn2")
        .def("isBF16", &QuantizedOminiFluxModel::isBF16); // Checks if the model is configured for bfloat16

    // Submodule for exposing specific C++ operators (kernels) to Python.
    m.def_submodule("ops")
        // Exposes a 4-bit weight, 4-bit activation GEMM kernel.
        .def("gemm_w4a4", nunchaku::ops::gemm_w4a4)
        // Exposes an FP16 attention kernel.
        .def("attention_fp16", nunchaku::ops::attention_fp16)
        // Exposes an AWQ (Activation-aware Weight Quantization) GEMM kernel.
        .def("gemm_awq", nunchaku::ops::gemm_awq)
        // Exposes an AWQ GEMV (Matrix-Vector multiplication) kernel.
        .def("gemv_awq", nunchaku::ops::gemv_awq)

        // Test functions for specific kernels, likely for internal validation.
        .def("test_rmsnorm_rope", nunchaku::ops::test_rmsnorm_rope)
        .def("test_pack_qkv", nunchaku::ops::test_pack_qkv);

    // Submodule for exposing utility functions.
    m.def_submodule("utils")
        // Sets the logging level for the C++ spdlog library.
        .def("set_log_level", [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); })
        // Sets CUDA stack limit.
        .def("set_cuda_stack_limit", nunchaku::utils::set_cuda_stack_limit)
        // Disables automatic memory release by the C++ backend (potentially for performance tuning).
        .def("disable_memory_auto_release", nunchaku::utils::disable_memory_auto_release)
        // Triggers memory trimming for the CUDA device.
        .def("trim_memory", nunchaku::utils::trim_memory)
        // Sets a mode for faster int-to-float conversions (potentially less precise but faster).
        .def("set_faster_i2f_mode", nunchaku::utils::set_faster_i2f_mode);
}
