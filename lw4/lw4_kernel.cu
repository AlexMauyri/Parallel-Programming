__global__ void vectors_sum_kernel(int *sum_vec, int *vec1, int *vec2, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        sum_vec[i] = vec1[i] + vec2[i];
    }
}

#include "lw4.c"