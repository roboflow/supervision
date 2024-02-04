#include <torch/extension.h>
#include <iostream>
#include <set>

at::Tensor build_adjacency_matrix_forward(torch::Tensor initial_rank);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor build_adjacency_matrix(at::Tensor initial_rank) {
    CHECK_INPUT(initial_rank);
    return build_adjacency_matrix_forward(initial_rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &build_adjacency_matrix, "build_adjacency_matrix (CUDA)");
}
