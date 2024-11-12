#include "gemm.h"
#include "flux.h"

#include <pybind11/pybind11.h>

// TORCH_LIBRARY(diffuxer, m) {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<QuantizedFluxModel>(m, "QuantizedFluxModel")
        // .def(torch::init<>())
        .def(py::init<>())
        .def("init", &QuantizedFluxModel::init,
            py::arg("bf16"),
            py::arg("deviceId")
        )
        .def("reset", &QuantizedFluxModel::reset)
        .def("load", &QuantizedFluxModel::load, 
            py::arg("path"),
            py::arg("partial") = false
        )
        .def("forward", &QuantizedFluxModel::forward)
        .def("forward_layer", &QuantizedFluxModel::forward_layer)
        .def("forward_single_layer", &QuantizedFluxModel::forward_single_layer)
        .def("disableMemoryAutoRelease", &QuantizedFluxModel::disableMemoryAutoRelease)
        .def("trimMemory", &QuantizedFluxModel::trimMemory)
        .def("startDebug", &QuantizedFluxModel::startDebug)
        .def("stopDebug", &QuantizedFluxModel::stopDebug)
        .def("getDebugResults", &QuantizedFluxModel::getDebugResults)
        .def("setLoraScale", &QuantizedFluxModel::setLoraScale)
        .def("forceFP16Attention", &QuantizedFluxModel::forceFP16Attention)
    ;
    py::class_<QuantizedGEMM>(m, "QuantizedGEMM")
        // .def(torch::init<>())
        .def(py::init<>())
        .def("init", &QuantizedGEMM::init)
        .def("reset", &QuantizedGEMM::reset)
        .def("load", &QuantizedGEMM::load)
        .def("forward", &QuantizedGEMM::forward)
        .def("quantize", &QuantizedGEMM::quantize)
        .def("gemm", &QuantizedGEMM::gemm)
        .def("gemv_awq", &QuantizedGEMM::gemv_awq)
        .def("startDebug", &QuantizedGEMM::startDebug)
        .def("stopDebug", &QuantizedGEMM::stopDebug)
        .def("getDebugResults", &QuantizedGEMM::getDebugResults)
    ;
}
