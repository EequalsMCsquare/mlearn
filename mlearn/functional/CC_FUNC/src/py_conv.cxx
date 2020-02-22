#include "conv.cxx"

#include <vector>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

py::array_t<double> batch_conv2d(py::array_t<double> &x, py::array_t<double> &w, std::vector<int> &out_shape){
    py::buffer_info x_buffer = x.request();
    py::buffer_info w_buffer = w.request();


    unsigned int out_size =  1;
    for(auto i = out_shape.begin(); i != out_shape.end(); i++)
        out_size *= *i;

    auto r = py::array_t<double>(out_size);
    py::buffer_info r_buffer = r.request();

    double  *ptr1 = (double *)r_buffer.ptr,
            *ptr2 = (double *)x_buffer.ptr,
            *ptr3 = (double *)w_buffer.ptr;

    std::size_t out_dim = w_buffer.shape[0];
    std::size_t *in_shape = new std::size_t[6];

    for(int i = 0; i < 6; i++)
        in_shape[i] = x_buffer.shape[i];
    batchConv2d(ptr1, ptr2, ptr3, in_shape, out_dim);
    delete[] in_shape;
    return r;
}

PYBIND11_MODULE(py_conv, m){
    m.doc() = "二维卷积 C Fucntion";
    m.def("batch_conv2d", &batch_conv2d,
        py::arg("inputs"),py::arg("weights"),py::arg("out_shape"));
}