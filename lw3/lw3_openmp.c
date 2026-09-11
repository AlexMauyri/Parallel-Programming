#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX 1020000
#define NUM_OF_THREADS 8
#define ITER_NUM 20
#define Q 20
#define EPSILON 1e-9

typedef struct {
    double *vec_1;
    double *vec_2;
    double *vec_3;
    size_t size;
} vectors;

typedef struct {
    double *sum_sequential;
    double *sum_static;
    double *sum_dynamic;
    double *sum_guided;
    size_t size;
} sum_vectors;

typedef struct {
    double time_sequential;
    double time_initialization;
    double time_static;
    double time_dynamic;
    double time_guided;
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

int vectors_malloc(vectors *v, size_t size) {
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

void sum_vectors_free(sum_vectors *sums) {
    free(sums->sum_sequential);
    free(sums->sum_static);
    free(sums->sum_dynamic);
    free(sums->sum_guided);
    sums->sum_sequential = NULL;
    sums->sum_static = NULL;
    sums->sum_dynamic = NULL;
    sums->sum_guided = NULL;
    sums->size = 0;
}

int sum_vectors_malloc(sum_vectors *sums, size_t size) {
    sums->sum_sequential = malloc(sizeof(double) * size);
    sums->sum_static = malloc(sizeof(double) * size);
    sums->sum_dynamic = malloc(sizeof(double) * size);
    sums->sum_guided = malloc(sizeof(double) * size);
    sums->size = size;

    if (!sums->sum_sequential || !sums->sum_static || !sums->sum_dynamic || !sums->sum_guided) {
        sum_vectors_free(sums);
        return -1;
    }

    return 0;
}

int arrays_equal(const double *a, const double *b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > EPSILON) {
            return 0;
        }
    }

    return 1;
}

void print_initial_data() {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of threads: %d\n", NUM_OF_THREADS);
    printf("Actual number of threads: %d\n", omp_get_max_threads());
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");
}

void print_final_results(sum_vectors *sums, elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Static sum is equal to sequential sum: %s\n",
           arrays_equal(sums->sum_sequential, sums->sum_static, sums->size) ? "Yes" : "No");
    printf("Dynamic sum is equal to sequential sum: %s\n",
           arrays_equal(sums->sum_sequential, sums->sum_dynamic, sums->size) ? "Yes" : "No");
    printf("Guided sum is equal to sequential sum: %s\n\n",
           arrays_equal(sums->sum_sequential, sums->sum_guided, sums->size) ? "Yes" : "No");

    printf("Sequential time = %.4f ms\n", stats->time_sequential * 1000);
    printf("Initialization time = %.4f ms\n", stats->time_initialization * 1000);
    printf("Static time = %.4f ms\n", stats->time_static * 1000);
    printf("Dynamic time = %.4f ms\n", stats->time_dynamic * 1000);
    printf("Guided time = %.4f ms\n\n", stats->time_guided * 1000);

    printf("With initialization:\n");
    printf("Static acceleration: %.4f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_static));
    printf("Dynamic acceleration: %.4f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_dynamic));
    printf("Guided acceleration: %.4f\n\n",
           stats->time_sequential / (stats->time_initialization + stats->time_guided));

    printf("Without initialization:\n");
    printf("Static acceleration: %.4f\n", stats->time_sequential / stats->time_static);
    printf("Dynamic acceleration: %.4f\n", stats->time_sequential / stats->time_dynamic);
    printf("Guided acceleration: %.4f\n", stats->time_sequential / stats->time_guided);
    printf("=====================================================\n");
}

void sum_sequential(vectors *v, double *sum_sequential) {
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_sequential[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_static_for(vectors *v, double *sum_static) {
    #pragma omp for schedule(static)
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_static[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_dynamic_for(vectors *v, double *sum_dynamic) {
    #pragma omp for schedule(dynamic)
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_dynamic[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_guided_for(vectors *v, double *sum_guided) {
    #pragma omp for schedule(guided)
    for (size_t j = 0; j < v->size; ++j) {
        for (int q = 0; q < Q; ++q) {
            sum_guided[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void normalize_results(sum_vectors *sums, elapsed_time_stats *stats) {
    for (size_t j = 0; j < sums->size; ++j) {
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
    omp_set_num_threads(NUM_OF_THREADS);
    print_initial_data();

    vectors v = {NULL, NULL, NULL, 0};
    if (vectors_malloc(&v, NMAX)) {
        fprintf(stderr, "Error: memory not allocated for vectors!\n");
        return -1;
    }

    sum_vectors sums = {NULL, NULL, NULL, NULL, 0};
    if (sum_vectors_malloc(&sums, NMAX)) {
        fprintf(stderr, "Error: memory not allocated for sum vectors!\n");
        return -1;
    } 

    for (size_t j = 0; j < v.size; ++j) {
        v.vec_1[j] = 1.0;
        v.vec_2[j] = 1.0;
        v.vec_3[j] = 1.0;
    }

    elapsed_time_stats stats;
    double start_time;

    memset(&stats, 0, sizeof(stats));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(sums.sum_sequential, 0, sums.size * sizeof(double));
        memset(sums.sum_static, 0, sums.size * sizeof(double));
        memset(sums.sum_dynamic, 0, sums.size * sizeof(double));
        memset(sums.sum_guided, 0, sums.size * sizeof(double));

        start_time = omp_get_wtime();
        sum_sequential(&v, sums.sum_sequential);
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

    vectors_free(&v);
    sum_vectors_free(&sums);
    return 0;
}
