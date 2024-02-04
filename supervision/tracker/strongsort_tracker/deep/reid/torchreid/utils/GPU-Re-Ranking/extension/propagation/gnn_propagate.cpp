#include <torch/extension.h>
#include <iostream>
#include <set>

at::Tensor gnn_propagate_forward(at::Tensor A, at::Tensor initial_rank, at::Tensor S);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor gnn_propagate(at::Tensor A ,at::Tensor initial_rank, at::Tensor S) {
    CHECK_INPUT(A);
    CHECK_INPUT(initial_rank);
    CHECK_INPUT(S);
    return gnn_propagate_forward(A, initial_rank, S);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gnn_propagate, "gnn propagate (CUDA)");
}