#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define NMAX     12000000
#define ITER_NUM 20
#define Q        35

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

int main(int argc, char* argv[]) {
    int process_rank, process_num;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (process_rank == 0) {
        printf("====================== PARAMS =======================\n");
        printf("Number of vectors: 2\n");
        printf("Number of elements: %d\n", NMAX);
        printf("Number of processes: %d\n", process_num);
        printf("Number of iterations: %d\n", ITER_NUM);
        printf("Q: %d\n", Q);
        printf("=====================================================\n");
    }

    size_t vector_size = NMAX;

    int *vec_1, *vec_2;

    vec_1 = malloc(sizeof(int) * vector_size);
    vec_2 = malloc(sizeof(int) * vector_size);

    if (!vec_1 || !vec_2) {
        fprintf(stderr, "Error: memory not allocated for vectors in process %d\n", process_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (process_rank == 0) {
        for (size_t i = 0; i < vector_size; ++i) {
            vec_1[i] = 1;
            vec_2[i] = 1;
        }
    }

    const size_t k = vector_size / process_num;
    const size_t i1 = k * process_rank;
    const size_t i2 = process_rank == process_num - 1 ? vector_size : k * (process_rank + 1);

    int sum_sequential = 0, sum_pp = 0, sum_reduce = 0;
    double time_sequential = 0.0, time_broadcast = 0.0, time_pp = 0.0, time_reduce = 0.0;
    double start_time;

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        if (process_rank == 0) {
            sum_sequential = 0;
            start_time = MPI_Wtime();
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_sequential += vec_1[i] + vec_2[i];
                }
            }
            time_sequential += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);
        }

        start_time = MPI_Wtime();
        MPI_Bcast(vec_1, vector_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vec_2, vector_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        time_broadcast += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);

        start_time = MPI_Wtime();
        {
            int process_sum = 0;
            MPI_Status status;

            for (size_t i = i1; i < i2; ++i) {
                for (int q = 0; q < Q; ++q) {
                    process_sum += vec_1[i] + vec_2[i];
                }
            }

            if (process_rank == 0) {
                sum_pp = process_sum;
                for (int i = 1; i < process_num; ++i) {
                    int recv_val = 0;
                    MPI_Recv(&recv_val, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                    sum_pp += recv_val;
                }
            }
            else {
                MPI_Send(&process_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        time_pp += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);

        start_time = MPI_Wtime();
        {
            int process_sum = 0;

            for (size_t i = i1; i < i2; ++i) {
                for (int q = 0; q < Q; ++q) {
                    process_sum += vec_1[i] + vec_2[i];
                }
            }

            int reduce_result = 0;
            MPI_Reduce(&process_sum, &reduce_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            if (process_rank == 0) {
                sum_reduce = reduce_result;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        time_reduce += SECONDS_TO_MILLISECONDS(MPI_Wtime() - start_time);
    }

    if (process_rank == 0) {
        sum_sequential /= Q;
        sum_pp /= Q;
        sum_reduce /= Q;

        time_sequential /= ITER_NUM;
        time_broadcast /= ITER_NUM;
        time_pp /= ITER_NUM;
        time_reduce /= ITER_NUM;

        printf("====================== RESULTS ======================\n");
        printf("Sequential sum = %d\n", sum_sequential);
        printf("P2P sum = %d\n", sum_pp);
        printf("Reduce sum = %d\n\n", sum_reduce);

        printf("Sequential time = %.4f ms\n", time_sequential);
        printf("Broadcast time = %.4f ms\n", time_broadcast);
        printf("P2P time = %.4f ms\n", time_pp);
        printf("Reduce time = %.4f ms\n\n", time_reduce);

        printf("With broadcast:\n");
        printf("P2P acceleration: %.4f\n", time_sequential / (time_broadcast + time_pp));
        printf("Reduce acceleration: %.4f\n\n", time_sequential / (time_broadcast + time_reduce));

        printf("Without broadcast:\n");
        printf("P2P acceleration: %.4f\n", time_sequential / time_pp);
        printf("Reduce acceleration: %.4f\n", time_sequential / time_reduce);
        printf("=====================================================\n");
    }

    free(vec_1);
    free(vec_2);

    MPI_Finalize();
    return 0;
}