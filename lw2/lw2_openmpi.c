#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define NMAX     100000
#define ITER_NUM 20
#define Q        20

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

typedef struct {
    double *vec_1;
    double *vec_2;
    double *vec_3;
    size_t size;
} vectors;

typedef struct {
    double sum_sequential;
    double sum_pp;
    double sum_reduce;
} sum_results;

typedef struct {
    double time_sequential;
    double time_broadcast;
    double time_pp;
    double time_reduce;
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
    v->vec_1 = malloc(sizeof(double) * size);
    v->vec_2 = malloc(sizeof(double) * size);
    v->vec_3 = malloc(sizeof(double) * size);
    v->size = size;

    if (!v->vec_1 || !v->vec_2 || !v->vec_3) {
        vectors_free(v);
        return -1;
    }

    return 0;
}

void print_initial_data(void) {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");
}

void print_final_results(const sum_results *sums, const elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Sequential sum = %.2f\n", sums->sum_sequential);
    printf("P2P sum = %.2f\n", sums->sum_pp);
    printf("Reduce sum = %.2f\n\n", sums->sum_reduce);

    printf("Sequential time = %.4f ms\n", stats->time_sequential);
    printf("Broadcast time = %.4f ms\n", stats->time_broadcast);
    printf("P2P time = %.4f ms\n", stats->time_pp);
    printf("Reduce time = %.4f ms\n\n", stats->time_reduce);

    printf("With broadcast:\n");
    printf("P2P acceleration: %.4f\n",
           stats->time_sequential / (stats->time_broadcast + stats->time_pp));
    printf("Reduce acceleration: %.4f\n\n",
           stats->time_sequential / (stats->time_broadcast + stats->time_reduce));

    printf("Without broadcast:\n");
    printf("P2P acceleration: %.4f\n", stats->time_sequential / stats->time_pp);
    printf("Reduce acceleration: %.4f\n", stats->time_sequential / stats->time_reduce);
    printf("=====================================================\n");
}

void sum_sequential(const vectors *v, sum_results *sums) {
    sums->sum_sequential = 0.0;
    for (size_t i = 0; i < v->size; ++i) {
        for (int q = 0; q < Q; ++q) {
            sums->sum_sequential += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
        }
    }
}

void sum_pp(const vectors *v, const size_t i1, const size_t i2, sum_results *sums,
            const int process_rank, const int process_num) {
    double process_sum = 0.0;
    MPI_Status status;

    for (size_t i = i1; i < i2; ++i) {
        for (int q = 0; q < Q; ++q) {
            process_sum += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
        }
    }

    if (process_rank == 0) {
        sums->sum_pp = process_sum;
        for (int i = 1; i < process_num; ++i) {
            double recv_val = 0.0;
            MPI_Recv(&recv_val, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            sums->sum_pp += recv_val;
        }
    }
    else {
        MPI_Send(&process_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void sum_reduce(const vectors *v, const size_t i1, const size_t i2, sum_results *sums,
    const int process_rank) {
    double process_sum = 0.0;

    for (size_t i = i1; i < i2; ++i) {
        for (int q = 0; q < Q; ++q) {
            process_sum += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
        }
    }

    double reduce_result = 0.0;
    MPI_Reduce(&process_sum, &reduce_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        sums->sum_reduce = reduce_result;
    }
}

void normalize_results(sum_results *sums, elapsed_time_stats *stats) {
    sums->sum_sequential /= Q;
    sums->sum_pp /= Q;
    sums->sum_reduce /= Q;

    stats->time_sequential /= ITER_NUM;
    stats->time_broadcast /= ITER_NUM;
    stats->time_pp /= ITER_NUM;
    stats->time_reduce /= ITER_NUM;
}

int main(int argc, char* argv[]) {
    int process_rank, process_num;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (process_rank == 0) {
        print_initial_data();
    }

    const size_t vector_size = NMAX;

    vectors v;
    if (vectors_malloc(&v, vector_size)) {
        fprintf(stderr, "Error: memory not allocated for vectors in process %d\n", process_rank);
        return -1;
    }
    if (process_rank == 0) {
        for (size_t i = 0; i < v.size; ++i) {
            v.vec_1[i] = 1.0;
            v.vec_2[i] = 1.0;
            v.vec_3[i] = 1.0;
        }
    }

    const size_t k = v.size / process_num;
    const size_t i1 = k * process_rank;
    const size_t i2 = process_rank == process_num - 1? v.size : k * (process_rank + 1);
    
    double start_time;
    sum_results sums;
    elapsed_time_stats stats;
    memset(&stats, 0, sizeof(stats));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(&sums, 0, sizeof(sums));

        if (process_rank == 0) {
            start_time = MPI_Wtime();
            sum_sequential(&v, &sums);
            stats.time_sequential += MPI_Wtime() - start_time;
        }

        start_time = MPI_Wtime();
        MPI_Bcast(v.vec_1, v.size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(v.vec_2, v.size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(v.vec_3, v.size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        stats.time_broadcast += MPI_Wtime() - start_time;

        MPI_Barrier(MPI_COMM_WORLD);

        start_time = MPI_Wtime();
        sum_pp(&v, i1, i2, &sums, process_rank, process_num);
        stats.time_pp += MPI_Wtime() - start_time;

        MPI_Barrier(MPI_COMM_WORLD);

        start_time = MPI_Wtime();
        sum_reduce(&v, i1, i2, &sums, process_rank);
        stats.time_reduce += MPI_Wtime() - start_time;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (process_rank == 0) {
        normalize_results(&sums, &stats);
        print_final_results(&sums, &stats);
    }

    vectors_free(&v);
    MPI_Finalize();
    return 0;
}
