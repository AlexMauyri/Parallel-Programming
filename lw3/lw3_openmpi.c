#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define NMAX          12000000
#define ADDITION      7669
#define NMAX_NON_MUL  (NMAX + ADDITION)
#define ITER_NUM      20
#define Q             35

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

int main(int argc, char* argv[]) {
    int process_rank, process_num;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    size_t vector_size_mul = NMAX;
    size_t vector_size_non_mul = NMAX_NON_MUL;

    int block_size_mul = vector_size_mul / process_num;
    int block_size_non_mul = vector_size_non_mul / process_num + (process_rank < vector_size_non_mul % process_num ? 1 : 0);

    int *vec_1_mul = NULL, *vec_2_mul = NULL;
    int *vec_1_non_mul = NULL, *vec_2_non_mul = NULL;
    int *vec_1_buf_mul = NULL, *vec_2_buf_mul = NULL;
    int *vec_1_buf_non_mul = NULL, *vec_2_buf_non_mul = NULL;

    size_t sums_mul_size = NMAX;
    size_t sums_non_mul_size = NMAX_NON_MUL;
    int *sum_sequential_mul = NULL, *sum_parallel_mul = NULL;
    int *sum_sequential_non_mul = NULL, *sum_parallel_non_mul = NULL;
    int *sum_parallel_mul_buf = NULL, *sum_parallel_non_mul_buf = NULL;

    int *counts = NULL, *displs = NULL;

    if (process_rank == 0) {
        printf("====================== PARAMS =======================\n");
        printf("Number of vectors: 2\n");
        printf("Vector type: int\n");
        printf("Number of elements for mul: %d\n", NMAX);
        printf("Number of elements for non mul: %d\n", NMAX_NON_MUL);
        printf("Number of iterations: %d\n", ITER_NUM);
        printf("Number of processes: %d\n", process_num);
        printf("Q: %d\n", Q);
        printf("=====================================================\n");

        vec_1_mul = malloc(sizeof(int) * vector_size_mul);
        vec_2_mul = malloc(sizeof(int) * vector_size_mul);

        if (!vec_1_mul || !vec_2_mul) {
            fprintf(stderr, "Error: memory not allocated for mul vectors!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        vec_1_non_mul = malloc(sizeof(int) * vector_size_non_mul);
        vec_2_non_mul = malloc(sizeof(int) * vector_size_non_mul);

        if (!vec_1_non_mul || !vec_2_non_mul) {
            fprintf(stderr, "Error: memory not allocated for non mul vectors!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        sum_sequential_mul = malloc(sizeof(int) * sums_mul_size);
        sum_parallel_mul = malloc(sizeof(int) * sums_mul_size);

        if (!sum_sequential_mul || !sum_parallel_mul) {
            fprintf(stderr, "Error: memory not allocated for mul sum vectors!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        sum_sequential_non_mul = malloc(sizeof(int) * sums_non_mul_size);
        sum_parallel_non_mul = malloc(sizeof(int) * sums_non_mul_size);

        if (!sum_sequential_non_mul || !sum_parallel_non_mul) {
            fprintf(stderr, "Error: memory not allocated for non mul sum vectors!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        counts = malloc(sizeof(int) * process_num);
        displs = malloc(sizeof(int) * process_num);

        if (!counts || !displs) {
            fprintf(stderr, "Error: memory not allocated for counts/displs\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (size_t i = 0; i < vector_size_mul; ++i) {
            vec_1_mul[i] = 1;
            vec_2_mul[i] = 1;
        }

        for (size_t i = 0; i < vector_size_non_mul; ++i) {
            vec_1_non_mul[i] = 1;
            vec_2_non_mul[i] = 1;
        }

        int accum = 0;
        int base = vector_size_non_mul / process_num;
        int rem = vector_size_non_mul % process_num;
        for (int i = 0; i < process_num; ++i) {
            counts[i] = base + (i < rem ? 1 : 0);
            displs[i] = accum;
            accum += counts[i];
        }
    }

    vec_1_buf_mul = malloc(sizeof(int) * block_size_mul);
    vec_2_buf_mul = malloc(sizeof(int) * block_size_mul);

    if (!vec_1_buf_mul || !vec_2_buf_mul) {
        fprintf(stderr, "Error: memory not allocated for mul vectors buffer in process %d\n!", process_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    vec_1_buf_non_mul = malloc(sizeof(int) * block_size_non_mul);
    vec_2_buf_non_mul = malloc(sizeof(int) * block_size_non_mul);

    if (!vec_1_buf_non_mul || !vec_2_buf_non_mul) {
        fprintf(stderr, "Error: memory not allocated for non mul vectors buffer in process %d\n!", process_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    sum_parallel_mul_buf = malloc(sizeof(int) * block_size_mul);

    if (sum_parallel_mul_buf == NULL) {
        fprintf(stderr, "Error: memory not allocated for mul sum vector buffer in process %d\n!", process_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    sum_parallel_non_mul_buf = malloc(sizeof(int) * block_size_non_mul);

    if (sum_parallel_non_mul_buf == NULL) {
        fprintf(stderr, "Error: memory not allocated for non mul sum vector buffer in process %d\n!", process_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double time_sequential_mul = 0.0, time_sequential_non_mul = 0.0;
    double time_scatter_cast = 0.0, time_scatterv_cast = 0.0;
    double time_parallel_mul = 0.0, time_parallel_non_mul = 0.0;
    double start_time;

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        if (process_rank == 0) {
            memset(sum_sequential_mul, 0, sums_mul_size * sizeof(int));
            memset(sum_sequential_non_mul, 0, sums_non_mul_size * sizeof(int));
            memset(sum_parallel_mul, 0, sums_mul_size * sizeof(int));
            memset(sum_parallel_non_mul, 0, sums_non_mul_size * sizeof(int));
        }
        memset(sum_parallel_mul_buf, 0, block_size_mul * sizeof(int));
        memset(sum_parallel_non_mul_buf, 0, block_size_non_mul * sizeof(int));

        if (process_rank == 0) {
            start_time = MPI_Wtime();
            for (size_t i = 0; i < vector_size_mul; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_sequential_mul[i] += vec_1_mul[i] + vec_2_mul[i];
                }
            }
            time_sequential_mul += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);

            start_time = MPI_Wtime();
            for (size_t i = 0; i < vector_size_non_mul; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_sequential_non_mul[i] += vec_1_non_mul[i] + vec_2_non_mul[i];
                }
            }
            time_sequential_non_mul += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        start_time = MPI_Wtime();
        MPI_Scatter(vec_1_mul, block_size_mul, MPI_INT, vec_1_buf_mul, block_size_mul, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(vec_2_mul, block_size_mul, MPI_INT, vec_2_buf_mul, block_size_mul, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        time_scatter_cast += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);

        start_time = MPI_Wtime();
        for (int i = 0; i < block_size_mul; ++i) {
            for (int q = 0; q < Q; ++q) {
                sum_parallel_mul_buf[i] += vec_1_buf_mul[i] + vec_2_buf_mul[i];
            }
        }
        MPI_Gather(sum_parallel_mul_buf, block_size_mul, MPI_INT, sum_parallel_mul, block_size_mul, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        time_parallel_mul += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);

        start_time = MPI_Wtime();
        MPI_Scatterv(vec_1_non_mul, counts, displs, MPI_INT, vec_1_buf_non_mul, block_size_non_mul, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(vec_2_non_mul, counts, displs, MPI_INT, vec_2_buf_non_mul, block_size_non_mul, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        time_scatterv_cast += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);

        start_time = MPI_Wtime();
        for (int i = 0; i < block_size_non_mul; ++i) {
            for (int q = 0; q < Q; ++q) {
                sum_parallel_non_mul_buf[i] += vec_1_buf_non_mul[i] + vec_2_buf_non_mul[i];
            }
        }
        MPI_Gatherv(sum_parallel_non_mul_buf, block_size_non_mul, MPI_INT, sum_parallel_non_mul, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        time_parallel_non_mul += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);
    }

    if (process_rank == 0) {
        for (size_t i = 0; i < sums_mul_size; ++i) {
            sum_sequential_mul[i] /= Q;
            sum_parallel_mul[i] /= Q;
        }
        for (size_t i = 0; i < sums_non_mul_size; ++i) {
            sum_sequential_non_mul[i] /= Q;
            sum_parallel_non_mul[i] /= Q;
        }

        time_sequential_mul /= ITER_NUM;
        time_sequential_non_mul /= ITER_NUM;
        time_scatter_cast /= ITER_NUM;
        time_scatterv_cast /= ITER_NUM;
        time_parallel_mul /= ITER_NUM;
        time_parallel_non_mul /= ITER_NUM;

        int mul_equal = 1;
        for (size_t i = 0; i < sums_mul_size; ++i) {
            if (sum_sequential_mul[i] != sum_parallel_mul[i]) {
                mul_equal = 0;
                break;
            }
        }

        int non_mul_equal = 1;
        for (size_t i = 0; i < sums_non_mul_size; ++i) {
            if (sum_sequential_non_mul[i] != sum_parallel_non_mul[i]) {
                non_mul_equal = 0;
                break;
            }
        }

        printf("====================== RESULTS ======================\n");
        printf("Mul parallel sum is equal mul sequential sum: %s\n", mul_equal ? "Yes" : "No");
        printf("Non mul parallel sum is equal non mul sequential sum: %s\n\n", non_mul_equal ? "Yes" : "No");

        printf("Mul sequential time = %.4f ms\n", time_sequential_mul);
        printf("Non mul sequential time = %.4f ms\n", time_sequential_non_mul);
        printf("Scatter cast time = %.4f ms\n", time_scatter_cast);
        printf("Scatterv time = %.4f ms\n", time_scatterv_cast);
        printf("Mul parallel time = %.4f ms\n", time_parallel_mul);
        printf("Non mul parallel time = %.4f ms\n\n", time_parallel_non_mul);

        printf("With scatter:\n");
        printf("Mul parallel acceleration: %.4f\n", time_sequential_mul / (time_scatter_cast + time_parallel_mul));
        printf("Non mul acceleration: %.4f\n\n", time_sequential_non_mul / (time_scatterv_cast + time_parallel_non_mul));

        printf("Without scatter:\n");
        printf("Mul parallel acceleration: %.4f\n", time_sequential_mul / time_parallel_mul);
        printf("Non mul acceleration: %.4f\n", time_sequential_non_mul / time_parallel_non_mul);
        printf("=====================================================\n");

        free(vec_1_mul);
        free(vec_2_mul);
        free(vec_1_non_mul);
        free(vec_2_non_mul);
        free(sum_sequential_mul);
        free(sum_parallel_mul);
        free(sum_sequential_non_mul);
        free(sum_parallel_non_mul);
        free(counts);
        free(displs);
    }

    free(vec_1_buf_mul);
    free(vec_2_buf_mul);
    free(vec_1_buf_non_mul);
    free(vec_2_buf_non_mul);
    free(sum_parallel_mul_buf);
    free(sum_parallel_non_mul_buf);

    MPI_Finalize();
    return 0;
}