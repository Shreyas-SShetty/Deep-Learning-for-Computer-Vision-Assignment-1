#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"
#include "ops.h"
#include "layers.h"
#include "loss.h"
#include "optim.h"
#include "stats.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_backend_ext, m) {
    m.doc() = "Custom Deep Learning Framework Backend (GNR638)";

    /* ---------------- Tensor ---------------- */
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<float>&,
                      const std::vector<int>&,
                      bool>(),
             py::arg("data"),
             py::arg("shape"),
             py::arg("requires_grad") = false)
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("requires_grad", &Tensor::requires_grad);

    /* ---------------- Ops ---------------- */
    m.def("add", &add, "Elementwise add");
    m.def("mul", &mul, "Elementwise multiply");
    m.def("matmul", &matmul, "Matrix multiplication");
    m.def("sum", &sum, "Sum reduction");
    m.def("mean", &mean, "Mean reduction");
    m.def("reshape", &reshape, "Reshape tensor");
    m.def("flatten", &flatten, "Flatten tensor");

    /* ---------------- Layers ---------------- */
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def_readwrite("weight", &Linear::weight)
        .def_readwrite("bias", &Linear::bias);

    py::class_<ReLU>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward);

    py::class_<Conv2D>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0)
        .def("forward", &Conv2D::forward)
        .def_readwrite("weight", &Conv2D::weight)
        .def_readwrite("bias", &Conv2D::bias);

    py::class_<MaxPool2D>(m, "MaxPool2D")
        .def(py::init<int, int>())
        .def("forward", &MaxPool2D::forward);

    /* ---------------- Loss ---------------- */
    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward);

    /* ---------------- Optimizer ---------------- */
    py::class_<SGD>(m, "SGD")
        .def(py::init<float>())
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    /* ---------------- Stats ---------------- */
    py::class_<ModelStats>(m, "ModelStats")
        .def_readwrite("parameters", &ModelStats::parameters)
        .def_readwrite("macs", &ModelStats::macs)
        .def_readwrite("flops", &ModelStats::flops);

    m.def("linear_params", &linear_params);
    m.def("linear_macs", &linear_macs);
    m.def("conv2d_params", &conv2d_params);
    m.def("conv2d_macs", &conv2d_macs);
    m.def("finalize_stats", &finalize_stats);
}
