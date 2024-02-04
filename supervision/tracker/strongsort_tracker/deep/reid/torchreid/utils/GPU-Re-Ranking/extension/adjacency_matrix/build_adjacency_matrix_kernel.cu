#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)


__global__ void build_adjacency_matrix_kernel(float* initial_rank, float* A, const int total_num, const int topk, const int nthreads, const int all_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < all_num; i += stride) {
        int ii = i / topk;
        A[ii * total_num + int(initial_rank[i])] = float(1.0);
    }
}

at::Tensor build_adjacency_matrix_forward(at::Tensor initial_rank) {
    const auto total_num = initial_rank.size(0);
    const auto topk = initial_rank.size(1);
    const auto all_num = total_num * topk;
    auto A = torch::zeros({total_num, total_num}, at::device(initial_rank.device()).dtype(at::ScalarType::Float));

    const int threads = 1024;
    const int blocks = (all_num + threads - 1) / threads;

    build_adjacency_matrix_kernel<<<blocks, threads>>>(initial_rank.data_ptr<float>(), A.data_ptr<float>(), total_num, topk, threads, all_num);
    return A;

}
