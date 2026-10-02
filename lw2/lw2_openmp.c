#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX            100000
#define NUM_OF_THREADS  8
#define ITER_NUM        20
#define Q               20

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

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

static double g_sum_reduction = 0.0;

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

void sum_sequential(const vectors *v, sum_results *sums) {
    sums->sum_sequential = 0.0;
    for (size_t i = 0; i < v->size; ++i) {
        for (int q = 0; q < Q; ++q) {
            sums->sum_sequential += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
        }
    }
}

void sum_reduction_for(const vectors *v, sum_results *sums) {
    #pragma omp for reduction(+:g_sum_reduction)
    for (size_t i = 0; i < v->size; ++i) {
        for (int q = 0; q < Q; ++q) {
            g_sum_reduction += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
        }
    }
    #pragma omp single 
    {
        sums->sum_reduction = g_sum_reduction;
        g_sum_reduction = 0.0;
    }
}

void sum_critical_for(const vectors *v, sum_results *sums) {
    #pragma omp for
    for (size_t i = 0; i < v->size; ++i) {
        for (int q = 0; q < Q; ++q) {
            double val = v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
            #pragma omp critical
            {
                sums->sum_critical += val;
            }
        }
    }
}

void sum_atomic_for(const vectors *v, sum_results *sums) {
    #pragma omp for
    for (size_t i = 0; i < v->size; ++i) {
        for (int q = 0; q < Q; ++q) {
            double val = v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
            #pragma omp atomic
            sums->sum_atomic += val;
        }
    }
}

void print_initial_data(void) {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of threads: %d\n", NUM_OF_THREADS);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");
}

void print_final_results(const sum_results *sums, const elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Sequential sum = %.2f\n", sums->sum_sequential);
    printf("Reduction sum = %.2f\n", sums->sum_reduction);
    printf("Critical sum = %.2f\n", sums->sum_critical);
    printf("Atomic sum = %.2f\n\n", sums->sum_atomic);

    printf("Sequential time = %.4f ms\n", stats->time_sequential);
    printf("Initialization time = %.4f ms\n", stats->time_initialization);
    printf("Reduction time = %.4f ms\n", stats->time_reduction);
    printf("Critical time = %.4f ms\n", stats->time_critical);
    printf("Atomic time = %.4f ms\n\n", stats->time_atomic);

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

int main(void) {
    omp_set_num_threads(NUM_OF_THREADS);
    print_initial_data();

    const size_t vector_size = NMAX;

    vectors v;
    if (vectors_malloc(&v, vector_size)) {
        fprintf(stderr, "Error: memory not allocated for vectors\n");
        return -1;
    }
    for (size_t i = 0; i < v.size; ++i) {
        v.vec_1[i] = 1.0;
        v.vec_2[i] = 1.0;
        v.vec_3[i] = 1.0;
    }

    sum_results sums;
    elapsed_time_stats stats;
    double start_time;
    memset(&stats, 0, sizeof(stats));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(&sums, 0, sizeof(sums));

        start_time = omp_get_wtime();
        sum_sequential(&v, &sums);
        stats.time_sequential += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);

        start_time = omp_get_wtime();

        #pragma omp parallel shared(v, sums, stats, start_time)
        {
            #pragma omp barrier
            #pragma omp single
            {
                stats.time_initialization += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            sum_reduction_for(&v, &sums);

            #pragma omp single
            {
                stats.time_reduction += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            sum_critical_for(&v, &sums);

            #pragma omp single
            {
                stats.time_critical += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            sum_atomic_for(&v, &sums);

            #pragma omp single
            {
                stats.time_atomic += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
            }
        }
    }

    normalize_results(&sums, &stats);
    print_final_results(&sums, &stats);

    vectors_free(&v); 
    return 0;
}
