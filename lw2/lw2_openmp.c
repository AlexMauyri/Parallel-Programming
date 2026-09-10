#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX 100000
#define NUM_OF_THREADS 8
#define ITER_NUM 20
#define Q 20

typedef struct {
    double *vec_1;
    double *vec_2;
    double *vec_3;
    size_t size;
} vectors;

typedef struct {
    double sum_sequential;
    double sum_reduction;
    double sum_critical;
    double sum_atomic;
} sum_results;

typedef struct {
    double time_sequential;
    double time_initialization;
    double time_reduction;
    double time_critical;
    double time_atomic;
} elapsed_time_stats;

double g_sum_reduction = 0.0;

int vectors_init(vectors *v, size_t size) {
    v->vec_1 = malloc(sizeof(double) * size);
    v->vec_2 = malloc(sizeof(double) * size);
    v->vec_3 = malloc(sizeof(double) * size);
    v->size = size;

    if (!v->vec_1 || !v->vec_2 || !v->vec_3) {
        return -1;
    }

    return 0;
}

void print_initial_data() {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of threads: %d\n", NUM_OF_THREADS);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");
}

void print_final_results(sum_results *sums, elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Sequential sum = %.2f\n", sums->sum_sequential);
    printf("Reduction sum = %.2f\n", sums->sum_reduction);
    printf("Critical sum = %.2f\n", sums->sum_critical);
    printf("Atomic sum = %.2f\n\n", sums->sum_atomic);

    printf("Sequential time = %.4f ms\n", stats->time_sequential * 1000);
    printf("Initialization time = %.4f ms\n", stats->time_initialization * 1000);
    printf("Reduction time = %.4f ms\n", stats->time_reduction * 1000);
    printf("Critical time = %.4f ms\n", stats->time_critical * 1000);
    printf("Atomic time = %.4f ms\n\n", stats->time_atomic * 1000);

    printf("With initialization:\n");
    printf("Reduction acceleration: %.4f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_reduction));
    printf("Critical acceleration: %.4f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_critical));
    printf("Atomic acceleration: %.4f\n\n",
           stats->time_sequential / (stats->time_initialization + stats->time_atomic));

    printf("Without initialization:\n");
    printf("Reduction acceleration: %.4f\n", stats->time_sequential / stats->time_reduction);
    printf("Critical acceleration: %.4f\n", stats->time_sequential / stats->time_critical);
    printf("Atomic acceleration: %.4f\n", stats->time_sequential / stats->time_atomic);
    printf("=====================================================\n");
}

void sum_sequential_for(vectors *v, sum_results *sums) {
    sums->sum_sequential = 0.0;
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            sums->sum_sequential += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_reduction_for(vectors *v, sum_results *sums) {
    #pragma omp for collapse(2) reduction(+:g_sum_reduction)
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            g_sum_reduction += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
    #pragma omp single
    sums->sum_reduction = g_sum_reduction;
}

void sum_critical_for(vectors *v, sum_results *sums) {
    #pragma omp for collapse(2)
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            double val = v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
            #pragma omp critical
            {
                sums->sum_critical += val;
            }
        }
    }
}

void sum_atomic_for(vectors *v, sum_results *sums) {
    #pragma omp for collapse(2)
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            double val = v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
            #pragma omp atomic
            sums->sum_atomic += val;
        }
    }
}

void normalize_results(sum_results *sums, elapsed_time_stats *stats) {
    sums->sum_sequential /= Q;
    sums->sum_reduction /= Q;
    sums->sum_critical /= Q;
    sums->sum_atomic /= Q;

    stats->time_sequential /= ITER_NUM;
    stats->time_initialization /= ITER_NUM;
    stats->time_reduction /= ITER_NUM;
    stats->time_critical /= ITER_NUM;
    stats->time_atomic /= ITER_NUM;
}

int main(int argc, char* argv[]) {
    print_initial_data();

    omp_set_num_threads(NUM_OF_THREADS);

    vectors v;
    if (vectors_init(&v, NMAX)) {
        fprintf(stderr, "Error: memory not allocated for vectors\n");
        return -1;
    }

    for (size_t j = 0; j < v.size; ++j) {
        v.vec_1[j] = 1.0;
        v.vec_2[j] = 1.0;
        v.vec_3[j] = 1.0;
    }

    sum_results sums;
    elapsed_time_stats stats;
    double start_time;

    memset(&stats, 0, sizeof(stats));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(&sums, 0, sizeof(sums));
        g_sum_reduction = 0.0;

        start_time = omp_get_wtime();
        sum_sequential_for(&v, &sums);
        stats.time_sequential += omp_get_wtime() - start_time;

        start_time = omp_get_wtime();

        #pragma omp parallel shared(v, sums, stats, start_time)
        {
            #pragma omp barrier
            #pragma omp single
            {
                stats.time_initialization += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_reduction_for(&v, &sums);

            #pragma omp single
            {
                stats.time_reduction += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_critical_for(&v, &sums);

            #pragma omp single
            {
                stats.time_critical += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_atomic_for(&v, &sums);

            #pragma omp single
            {
                stats.time_atomic += omp_get_wtime() - start_time;
            }
        }
    }

    normalize_results(&sums, &stats);
    print_final_results(&sums, &stats);

    free(v.vec_1);
    free(v.vec_2);
    free(v.vec_3);

    return 0;
}
