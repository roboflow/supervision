#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void gnn_propagate_forward_kernel(float* initial_rank, float* A, float* A_qe, float* S, const int sample_num, const int topk, const int total_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < total_num; i += stride) {
        int fea = i % sample_num;
        int sample_index = i / sample_num;
        float sum = 0.0;
        for (int j = 0; j < topk ; j++) {
            int topk_fea_index = int(initial_rank[sample_index*topk+j]) * sample_num + fea;
            sum += A[ topk_fea_index] * S[sample_index*topk+j];
        }
        A_qe[i] = sum;
    }
}

at::Tensor gnn_propagate_forward(at::Tensor A, at::Tensor initial_rank, at::Tensor S) {
    const auto sample_num = A.size(0);
    const auto topk = initial_rank.size(1);

    const auto total_num = sample_num * sample_num ; 
    auto A_qe = torch::zeros({sample_num, sample_num}, at::device(initial_rank.device()).dtype(at::ScalarType::Float));

    const int threads = 1024;
    const int blocks = (total_num + threads - 1) / threads;

    gnn_propagate_forward_kernel<<<blocks, threads>>>(initial_rank.data_ptr<float>(), A.data_ptr<float>(), A_qe.data_ptr<float>(), S.data_ptr<float>(), sample_num, topk, total_num);
    return A_qe;

}