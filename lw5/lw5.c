#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N                                   15
#define ITER_NUM                            20
#define FLOAT_PRECISION                     2
#define MATRIX_DISTANCE_BETWEEN_ELEMENTS    3
#define MATRIX_PRINT_HEAD                   3

#define CUDA_CALL(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if ((err) != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                 \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CUBLAS_CALL(call)                                                   \
    do {                                                                    \
        cublasStatus_t st = (call);                                         \
        if ((st) != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "CUBLAS error %d at %s:%d\n",                   \
                    (int)st, __FILE__, __LINE__);                           \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CURAND_CALL(call)                                                   \
    do {                                                                    \
        curandStatus_t st = (call);                                         \
        if ((st) != CURAND_STATUS_SUCCESS) {                                \
            fprintf(stderr, "CURAND error %d at %s:%d\n",                   \
                    (int)st, __FILE__, __LINE__);                           \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define PRINT_MATRIX(m)                                                     \
    do {                                                                    \
        printf("%s = [\n", #m);                                             \
        print_matrix(m);                                                    \
        printf("]\n");                                                      \
    } while (0)

typedef struct {
    float *items;
    size_t rows;
    size_t cols;
} matrix;

typedef struct {
    float time_device_matrix_creation;
    float time_host_matrix_creation;
    float time_sequential;
    float time_from_cpu_to_gpu;
    float time_parallel;
    float time_from_gpu_to_cpu;
} elapsed_time_stats;

void matrix_free(matrix *m) {
    free(m->items);
    m->items = NULL;
    m->rows = 0;
    m->cols = 0;
}

int matrix_malloc(matrix *m, const size_t rows, const size_t cols) {
    m->items = (float*)malloc(sizeof(float) * rows * cols);
    m->rows = rows;
    m->cols = cols;

    if (!m->items) {
        matrix_free(m);
        fprintf(stderr, "Error: memory not allocated for matrix\n");
        return -1;
    }
    return 0;
}

void matrix_cuda_free(matrix *m) {
    if (m->items) cudaFree(m->items);
    m->items = NULL;
    m->rows = 0;
    m->cols = 0;
}

int matrix_cuda_malloc(matrix *m, const size_t rows, const size_t cols) {
    m->items = NULL;
    m->rows = 0;
    m->cols = 0;

    CUDA_CALL(cudaMalloc((void**)&(m->items), sizeof(float) * rows * cols));
    m->rows = rows;
    m->cols = cols;
    return 0;
}

void matrix_fill_rand(const matrix *m) {
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            m->items[i * m->cols + j] = (double)rand() / RAND_MAX;
        }
    }
}

void matrix_multiplication(const matrix *A, const matrix *B, matrix *C) {
    for (size_t i = 0; i < A->rows; ++i) {
        for (size_t j = 0; j < B->cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A->cols; ++k) {
                sum += A->items[i * A->cols + k] * B->items[k * B->cols + j];
            }
            C->items[i * C->cols + j] = sum;
        }
    }
}

void print_matrix(const matrix m) {
    const int width = FLOAT_PRECISION + MATRIX_DISTANCE_BETWEEN_ELEMENTS;

    const int row_dots = (m.rows > 2 * MATRIX_PRINT_HEAD);
    const int col_dots = (m.cols > 2 * MATRIX_PRINT_HEAD);

    const size_t top    = row_dots ? MATRIX_PRINT_HEAD : m.rows;
    const size_t bottom = row_dots ? MATRIX_PRINT_HEAD : 0;
    const size_t left   = col_dots ? MATRIX_PRINT_HEAD : m.cols;
    const size_t right  = col_dots ? MATRIX_PRINT_HEAD : 0;

    const size_t printed_rows = top + bottom + (row_dots ? 1 : 0);
    const size_t printed_cols = left + right + (col_dots ? 1 : 0);

    for (size_t ri = 0; ri < printed_rows; ++ri) {
        if (row_dots && ri == top) {
            for (size_t c = 0; c < printed_cols; ++c) {
                printf("%*s ", width, "...");
            }
            printf("\n");
            continue;
        }

        const size_t i = (ri < top) ? ri : (m.rows - bottom + (ri - top - 1));

        for (size_t j = 0; j < left; ++j) {
            printf("%*.*f ", width, FLOAT_PRECISION, m.items[i * m.cols + j]);
        }

        if (col_dots) {
            printf("%*s ", width, "...");
        }

        for (size_t k = 0; k < right; ++k) {
            const size_t j = m.cols - right + k;
            printf("%*.*f ", width, FLOAT_PRECISION, m.items[i * m.cols + j]);
        }

        printf("\n");
    }
}

int GPU_fill_rand(const matrix *m) {
    curandGenerator_t prng;
    CURAND_CALL(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock()));
    CURAND_CALL(curandGenerateUniform(prng, m->items, m->rows * m->cols));
    CURAND_CALL(curandDestroyGenerator(prng));
    return 0;
}

int GPU_blas_mmul(const matrix *A, const matrix *B, matrix *C) {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)C->cols, (int)C->rows, (int)A->cols,
                            &alpha,
                            B->items, (int)B->cols,
                            A->items, (int)A->cols,
                            &beta,
                            C->items, (int)C->cols));

    CUBLAS_CALL(cublasDestroy(handle));
    return 0;
}

float elapsed_time_ms(const struct timespec start, const struct timespec stop) {
    return (float)(stop.tv_sec - start.tv_sec) * 1e3f + (float)(stop.tv_nsec - start.tv_nsec) * 1e-6f;
}

void print_initial_data(void) {
    printf("====================== PARAMS =======================\n");
    printf("N: %d\n", N);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("=====================================================\n");
}

void print_final_results(const elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Sequential time = %.4f ms\n", stats->time_sequential);
    printf("Device matrix creation time = %.4f ms\n", stats->time_device_matrix_creation);
    printf("Host matrix creation time = %.4f ms\n", stats->time_host_matrix_creation);
    printf("CPU->GPU transfer time = %.4f ms\n", stats->time_from_cpu_to_gpu);
    printf("Parallel time = %.4f ms\n", stats->time_parallel);
    printf("GPU->CPU transfer time = %.4f ms\n\n", stats->time_from_gpu_to_cpu);

    printf("Without data transfer:\n");
    printf("Multiplication acceleration: %.4f\n", stats->time_sequential / stats->time_parallel);
    printf("Generation acceleration: %.4f\n\n", stats->time_host_matrix_creation / stats->time_device_matrix_creation);

    float sequential_total = stats->time_host_matrix_creation + stats->time_sequential;

    float parallel_total_host = stats->time_host_matrix_creation +
                                stats->time_from_cpu_to_gpu +
                                stats->time_parallel +
                                stats->time_from_gpu_to_cpu;

    float parallel_total_device = stats->time_device_matrix_creation +
                                  stats->time_parallel +
                                  stats->time_from_gpu_to_cpu;

    printf("With data transfer:\n");
    printf("Final acceleration (host matrix creation + transfer + return): %.4f\n", sequential_total / parallel_total_host);

    printf("Final acceleration (device matrix creation + return): %.4f\n", sequential_total / parallel_total_device);
    printf("=====================================================\n");
}

void normalize_results(elapsed_time_stats *stats) {
    stats->time_device_matrix_creation /= ITER_NUM;
    stats->time_host_matrix_creation /= ITER_NUM;
    stats->time_sequential /= ITER_NUM;
    stats->time_from_cpu_to_gpu /= ITER_NUM;
    stats->time_parallel /= ITER_NUM;
    stats->time_from_gpu_to_cpu /= ITER_NUM;
}

int main(void) {
    print_initial_data();

    const size_t matrix_size = N;

    struct timespec start_seq, stop_seq;
    elapsed_time_stats stats;
    memset(&stats, 0, sizeof(stats));

    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        matrix A, B, C;

        if (matrix_malloc(&A, matrix_size, matrix_size) || 
            matrix_malloc(&B, matrix_size, matrix_size) || 
            matrix_malloc(&C, matrix_size, matrix_size)) {
            return -1;
        }

        clock_gettime(CLOCK_MONOTONIC, &start_seq);

        matrix_fill_rand(&A);
        matrix_fill_rand(&B);

        clock_gettime(CLOCK_MONOTONIC, &stop_seq);
        stats.time_host_matrix_creation += elapsed_time_ms(start_seq, stop_seq);

        matrix A_device, B_device, C_device;

        if (matrix_cuda_malloc(&A_device, matrix_size, matrix_size) || 
            matrix_cuda_malloc(&B_device, matrix_size, matrix_size) || 
            matrix_cuda_malloc(&C_device, matrix_size, matrix_size)) {
            return -1;
        }

        CUDA_CALL(cudaEventRecord(start, 0));

        if (GPU_fill_rand(&A_device) != 0 || GPU_fill_rand(&B_device) != 0) {
            return -1;
        }

        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_device_matrix_creation += gpu_time;

        if (iteration == ITER_NUM - 1) {
            PRINT_MATRIX(A);
            PRINT_MATRIX(B);
        }

        clock_gettime(CLOCK_MONOTONIC, &start_seq);

        matrix_multiplication(&A, &B, &C);

        clock_gettime(CLOCK_MONOTONIC, &stop_seq);
        stats.time_sequential += elapsed_time_ms(start_seq, stop_seq);

        if (iteration == ITER_NUM - 1) {
            printf("Result on CPU:\n");
            PRINT_MATRIX(C);
        }

        CUDA_CALL(cudaEventRecord(start, 0));

        CUDA_CALL(cudaMemcpy(A_device.items, A.items, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(B_device.items, B.items, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_from_cpu_to_gpu += gpu_time;

        CUDA_CALL(cudaEventRecord(start, 0));

        if (GPU_blas_mmul(&A_device, &B_device, &C_device) != 0) {
            return -1;
        }

        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_parallel += gpu_time;

        CUDA_CALL(cudaEventRecord(start, 0));

        CUDA_CALL(cudaMemcpy(C.items, C_device.items, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaEventRecord(stop, 0));
        CUDA_CALL(cudaEventSynchronize(stop));
        CUDA_CALL(cudaEventElapsedTime(&gpu_time, start, stop));
        stats.time_from_gpu_to_cpu += gpu_time;

        if (iteration == ITER_NUM - 1) {
            printf("Result on GPU:\n");
            PRINT_MATRIX(C);
        }

        matrix_free(&A);
        matrix_free(&B);
        matrix_free(&C);
        matrix_cuda_free(&A_device);
        matrix_cuda_free(&B_device);
        matrix_cuda_free(&C_device);
    }

    normalize_results(&stats);
    print_final_results(&stats);

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    return 0;
}