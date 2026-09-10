#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX 10200
#define NUM_OF_THREADS 8
#define ITER_NUM 20
#define Q 20

typedef struct {
    double *vec_1;
    double *vec_2;
    double *vec_3;
} vectors;

typedef struct {
    double *sum_sequential;
    double *sum_static;
    double *sum_dynamic;
    double *sum_guided;
} sum_arrays;

typedef struct {
    double time_sequential;
    double time_initialization;
    double time_static;
    double time_dynamic;
    double time_guided;
} elapsed_time_stats;

void print_initial_data() {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of threads: %d\n", NUM_OF_THREADS);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");
}

void print_final_results(sum_arrays *sums, elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Static sum is equal sequential sum: %s\n",
           memcmp(sums->sum_sequential, sums->sum_static, NMAX * sizeof(double)) ? "No" : "Yes");
    printf("Dynamic sum is equal sequential sum: %s\n",
           memcmp(sums->sum_sequential, sums->sum_dynamic, NMAX * sizeof(double)) ? "No" : "Yes");
    printf("Guided sum is equal sequential sum: %s\n\n",
           memcmp(sums->sum_sequential, sums->sum_guided, NMAX * sizeof(double)) ? "No" : "Yes");

    printf("Sequential time = %f\n", stats->time_sequential);
    printf("Initialization time = %f\n", stats->time_initialization);
    printf("Static time = %f\n", stats->time_static);
    printf("Dynamic time = %f\n", stats->time_dynamic);
    printf("Guided time = %f\n\n", stats->time_guided);

    printf("With initialization:\n");
    printf("Static acceleration: %f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_static));
    printf("Dynamic acceleration: %f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_dynamic));
    printf("Guided acceleration: %f\n\n",
           stats->time_sequential / (stats->time_initialization + stats->time_guided));

    printf("Without initialization:\n");
    printf("Static acceleration: %f\n", stats->time_sequential / stats->time_static);
    printf("Dynamic acceleration: %f\n", stats->time_sequential / stats->time_dynamic);
    printf("Guided acceleration: %f\n", stats->time_sequential / stats->time_guided);
    printf("=====================================================\n");
}

void sum_sequential_for(vectors *v, double *sum_sequential) {
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_sequential[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_static_for(vectors *v, double *sum_static) {
    #pragma omp for schedule(static)
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_static[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_dynamic_for(vectors *v, double *sum_dynamic) {
    #pragma omp for schedule(dynamic)
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_dynamic[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_guided_for(vectors *v, double *sum_guided) {
    #pragma omp for schedule(guided)
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_guided[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void normalize_results(sum_arrays *sums, elapsed_time_stats *stats) {
    for (int j = 0; j < NMAX; ++j) {
        sums->sum_sequential[j] /= Q;
        sums->sum_static[j] /= Q;
        sums->sum_dynamic[j] /= Q;
        sums->sum_guided[j] /= Q;
    }

    stats->time_sequential /= ITER_NUM;
    stats->time_initialization /= ITER_NUM;
    stats->time_static /= ITER_NUM;
    stats->time_dynamic /= ITER_NUM;
    stats->time_guided /= ITER_NUM;
}

int main(int argc, char* argv[]) {
    print_initial_data();

    omp_set_num_threads(NUM_OF_THREADS);

    vectors v;
    v.vec_1 = malloc(sizeof(double) * NMAX);
    v.vec_2 = malloc(sizeof(double) * NMAX);
    v.vec_3 = malloc(sizeof(double) * NMAX);

    sum_arrays sums;
    sums.sum_sequential = malloc(sizeof(double) * NMAX);
    sums.sum_static = malloc(sizeof(double) * NMAX);
    sums.sum_dynamic = malloc(sizeof(double) * NMAX);
    sums.sum_guided = malloc(sizeof(double) * NMAX);

    for (int j = 0; j < NMAX; ++j) {
        v.vec_1[j] = 1.0;
        v.vec_2[j] = 1.0;
        v.vec_3[j] = 1.0;
    }

    elapsed_time_stats stats;
    double start_time;

    memset(&stats, 0, sizeof(stats));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(sums.sum_sequential, 0, NMAX * sizeof(double));
        memset(sums.sum_static, 0, NMAX * sizeof(double));
        memset(sums.sum_dynamic, 0, NMAX * sizeof(double));
        memset(sums.sum_guided, 0, NMAX * sizeof(double));

        start_time = omp_get_wtime();
        sum_sequential_for(&v, sums.sum_sequential);
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

            sum_static_for(&v, sums.sum_static);

            #pragma omp single
            {
                stats.time_static += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_dynamic_for(&v, sums.sum_dynamic);

            #pragma omp single
            {
                stats.time_dynamic += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_guided_for(&v, sums.sum_guided);

            #pragma omp single
            {
                stats.time_guided += omp_get_wtime() - start_time;
            }
        }
    }

    normalize_results(&sums, &stats);
    print_final_results(&sums, &stats);

    free(v.vec_1);
    free(v.vec_2);
    free(v.vec_3);

    free(sums.sum_sequential);
    free(sums.sum_static);
    free(sums.sum_dynamic);
    free(sums.sum_guided);

    return 0;
}
