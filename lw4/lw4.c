#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define NMAX        1000000
#define GRID_SIZE   512
#define BLOCK_SIZE  512
#define ITER_NUM    20

#define CUDA_CALL(call)                                                      \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if ((err) != cudaSuccess) {                                          \
            fprintf(stderr, "Error \"%s\" at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            return -1;                                                       \
        }                                                                    \
    } while (0)

#define ELAPSED_TIME_MS(start, end) \
    ((float)((end).tv_sec - (start).tv_sec) * 1e3f + \
     (float)((end).tv_nsec - (start).tv_nsec) * 1e-6f)

int main(void) {
    const int vector_size = NMAX;
    const int bytes = vector_size * sizeof(int);

    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 2\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("=====================================================\n");

    int *h_vec_1 = (int*)malloc(bytes);
    int *h_vec_2 = (int*)malloc(bytes);
    int *h_sum_seq = (int*)malloc(bytes);
    int *h_sum_par = (int*)malloc(bytes);

    if (!h_vec_1 || !h_vec_2 || !h_sum_seq || !h_sum_par) {
        fprintf(stderr, "Error: Host memory not allocated!\n");
        return -1;
    }

    for (int i = 0; i < vector_size; ++i) {
        h_vec_1[i] = 1;
        h_vec_2[i] = 1;
    }

    int *d_vec_1 = NULL, *d_vec_2 = NULL, *d_sum = NULL;
    CUDA_CALL(cudaMalloc((void**)&d_vec_1, bytes));
    CUDA_CALL(cudaMalloc((void**)&d_vec_2, bytes));
    CUDA_CALL(cudaMalloc((void**)&d_sum, bytes));

    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    struct timespec start_seq, stop_seq;
    float time_sequential = 0.0f;
    float time_gpu_transfer = 0.0f;
    float time_parallel = 0.0f;

    dim3 blocks = dim3(GRID_SIZE, 1);
    dim3 threads = dim3(BLOCK_SIZE, 1);

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(h_sum_seq, 0, bytes);
        memset(h_sum_par, 0, bytes);
        CUDA_CALL(cudaMemset(d_sum, 0, bytes));

        clock_gettime(CLOCK_MONOTONIC, &start_seq);
        for (int i = 0; i < vector_size; ++i) {
            h_sum_seq[i] += h_vec_1[i] + h_vec_2[i];
        }
        clock_gettime(CLOCK_MONOTONIC, &stop_seq);
        time_sequential += ELAPSED_TIME_MS(start_seq, stop_seq);

        CUDA_CALL(cudaEventRecord(start, 0));
        CUDA_CALL(cudaMemcpy(d_vec_1, h_vec_1, bytes, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_vec_2, h_vec_2, bytes, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        time_gpu_transfer += gpu_time;

        CUDA_CALL(cudaEventRecord(start, 0));
        vectors_sum_kernel<<<blocks, threads>>>(d_sum, d_vec_1, d_vec_2, vector_size);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        time_parallel += gpu_time;

        CUDA_CALL(cudaEventRecord(start, 0));
        CUDA_CALL(cudaMemcpy(h_sum_par, d_sum, bytes, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        time_gpu_transfer += gpu_time;
    }

    time_sequential /= ITER_NUM;
    time_gpu_transfer /= ITER_NUM;
    time_parallel /= ITER_NUM;

    int is_equal = 1;
    for (int i = 0; i < vector_size; ++i) {
        if (h_sum_seq[i] != h_sum_par[i]) {
            is_equal = 0;
            break;
        }
    }

    printf("====================== RESULTS ======================\n");
    printf("GPU parallel sum is equal to sequential sum: %s\n\n", is_equal ? "Yes" : "No");
    printf("Sequential time = %.4f ms\n", time_sequential);
    printf("GPU broadcasting time = %.4f ms\n", time_gpu_transfer);
    printf("Parallel time = %.4f ms\n\n", time_parallel);
    printf("With GPU transfer:\n");
    printf("Parallel acceleration: %.4f\n\n", time_sequential / (time_gpu_transfer + time_parallel));
    printf("Without GPU transfer:\n");
    printf("Parallel acceleration: %.4f\n\n", time_sequential / time_parallel);
    printf("=====================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_vec_1);
    cudaFree(d_vec_2);
    cudaFree(d_sum);
    free(h_vec_1);
    free(h_vec_2);
    free(h_sum_seq);
    free(h_sum_par);

    return 0;
}