#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define NMAX        100000
#define BLOCK_SIZE  512
#define ITER_NUM    20
#define EPSILON     1e-9

#define CUDA_CALL(call)                                                      \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if ((err) != cudaSuccess) {                                          \
            fprintf(stderr, "Error \"%s\" at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    double *vec_1;
    double *vec_2;
    double *vec_3;
    size_t size;
} vectors;

typedef struct {
    double *sum_sequential;
    double *sum_parallel;
    size_t size;
} sum_vectors;

typedef struct {
    float time_sequential;
    float time_gpu_transfer;
    float time_parallel;
} elapsed_time_stats;

void vectors_free(vectors *v) {
    free(v->vec_1);
    free(v->vec_2);
    free(v->vec_3);
    v->vec_1 = NULL;
    v->vec_2 = NULL;
    v->vec_3 = NULL;
    v->size = 0;
}

int vectors_malloc(vectors *v, const size_t size) {
    v->vec_1 = (double*)malloc(sizeof(double) * size);
    v->vec_2 = (double*)malloc(sizeof(double) * size);
    v->vec_3 = (double*)malloc(sizeof(double) * size);
    v->size = 0;

    if (!v->vec_1 || !v->vec_2 || !v->vec_3) {
        vectors_free(v);
        return -1;
    }

    v->size = size;
    return 0;
}

void vectors_cuda_free(vectors *v) {
    cudaFree(v->vec_1);
    cudaFree(v->vec_2);
    cudaFree(v->vec_3);
    v->vec_1 = NULL;
    v->vec_2 = NULL;
    v->vec_3 = NULL;
    v->size = 0;
}

int vectors_cuda_malloc(vectors *v, const size_t size) {
    v->vec_1 = NULL;
    v->vec_2 = NULL;
    v->vec_3 = NULL;
    v->size = 0;

    CUDA_CALL(cudaMalloc((void**)&(v->vec_1), size * sizeof(double)));
    CUDA_CALL(cudaMalloc((void**)&(v->vec_2), size * sizeof(double)));
    CUDA_CALL(cudaMalloc((void**)&(v->vec_3), size * sizeof(double)));

    v->size = size;
    return 0;
}

void vectors_sum(const vectors *v, double *sum_result) {
    for (size_t i = 0; i < v->size; ++i) {
        sum_result[i] += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
    }
}

void sum_vectors_free(sum_vectors *sums) {
    free(sums->sum_sequential);
    free(sums->sum_parallel);
    sums->sum_sequential = NULL;
    sums->sum_parallel = NULL;
    sums->size = 0;
}

int sum_vectors_malloc(sum_vectors *sums, const size_t size) {
    sums->sum_sequential = (double*)malloc(sizeof(double) * size);
    sums->sum_parallel = (double*)malloc(sizeof(double) * size);
    sums->size = size;

    if (!sums->sum_sequential || !sums->sum_parallel) {
        sum_vectors_free(sums);
        return -1;
    }

    return 0;
}

int arrays_equal(const double *a, const double *b, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > EPSILON) {
            return 0;
        }
    }
    return 1;
}

float elapsed_time_ms(const struct timespec start, const struct timespec end) {
    return (float)(end.tv_sec - start.tv_sec) * 1e3f
         + (float)(end.tv_nsec - start.tv_nsec) * 1e-6f;
}

void print_initial_data(void) {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("=====================================================\n");
}

void print_final_results(const sum_vectors *sums, const elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("GPU parallel sum is equal to sequential sum: %s\n\n",
            arrays_equal(sums->sum_sequential, sums->sum_parallel, sums->size) ? "Yes" : "No");

    printf("Sequential time = %.4f ms\n", stats->time_sequential);
    printf("GPU broadcasting time = %.4f ms\n", stats->time_gpu_transfer);
    printf("Parallel time = %.4f ms\n\n", stats->time_parallel);

    printf("With GPU transfer:\n");
    printf("Parallel acceleration: %.4f\n\n",
            stats->time_sequential / (stats->time_gpu_transfer + stats->time_parallel));

    printf("Without GPU transfer:\n");
    printf("Parallel acceleration: %.4f\n\n",
            stats->time_sequential / stats->time_parallel);
    printf("=====================================================\n");
}

void normalize_results(elapsed_time_stats *stats) {
    stats->time_sequential /= ITER_NUM;
    stats->time_gpu_transfer /= ITER_NUM;
    stats->time_parallel /= ITER_NUM;
}

int main(void) {
    print_initial_data();

    const size_t vector_size = NMAX;

    vectors v = {NULL, NULL, NULL, 0};
    if (vectors_malloc(&v, vector_size)) {
        fprintf(stderr, "Error: memory not allocated for vectors!\n");
        return -1;
    }
    for (size_t i = 0; i < v.size; ++i) {
        v.vec_1[i] = 1.0;
        v.vec_2[i] = 1.0;
        v.vec_3[i] = 1.0;
    }

    sum_vectors sums = {NULL, NULL, 0};
    if (sum_vectors_malloc(&sums, vector_size)) {
        fprintf(stderr, "Error: memory not allocated for sum vectors!\n");
        return -1;
    }

    vectors v_device = {NULL, NULL, NULL, 0};
    if (vectors_cuda_malloc(&v_device, vector_size) != 0) {
        return -1;
    }

    double *sum_device = NULL;
    CUDA_CALL(cudaMalloc((void**)&sum_device, vector_size * sizeof(double)));

    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    struct timespec start_seq, stop_seq;
    elapsed_time_stats stats;
    memset(&stats, 0, sizeof(stats));

    dim3 blocks  = dim3((vector_size + BLOCK_SIZE - 1) / BLOCK_SIZE,  1);
    dim3 threads = dim3(BLOCK_SIZE, 1);

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(sums.sum_sequential, 0, sums.size * sizeof(*sums.sum_sequential));
        memset(sums.sum_parallel,   0, sums.size * sizeof(*sums.sum_parallel));
        CUDA_CALL(cudaMemset(sum_device, 0, sums.size * sizeof(*sums.sum_parallel)));

        clock_gettime(CLOCK_MONOTONIC, &start_seq);

        vectors_sum(&v, sums.sum_sequential);

        clock_gettime(CLOCK_MONOTONIC, &stop_seq);
        stats.time_sequential += elapsed_time_ms(start_seq, stop_seq);

        CUDA_CALL(cudaEventRecord(start, 0));

        CUDA_CALL(cudaMemcpy(v_device.vec_1, v.vec_1, v.size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(v_device.vec_2, v.vec_2, v.size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(v_device.vec_3, v.vec_3, v.size * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_gpu_transfer += gpu_time;

        CUDA_CALL(cudaEventRecord(start, 0));

        vectors_sum_kernel<<<blocks, threads>>>(sum_device, v_device.vec_1, v_device.vec_2, v_device.vec_3, v_device.size);

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_parallel += gpu_time;

        CUDA_CALL(cudaEventRecord(start, 0));

        CUDA_CALL(cudaMemcpy(sums.sum_parallel, sum_device, sums.size * sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_gpu_transfer += gpu_time;
    }

    normalize_results(&stats);
    print_final_results(&sums, &stats);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectors_cuda_free(&v_device);
    cudaFree(sum_device);
    vectors_free(&v);
    sum_vectors_free(&sums);

    return 0;
}