__global__ void vectors_sum_kernel(double *sum_vec, double *vec1, double *vec2, double *vec3, unsigned long long size) {
  unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) sum_vec[idx] = vec1[idx] + vec2[idx] + vec3[idx];
}

#include "lw4.c"