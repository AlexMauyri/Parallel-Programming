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
} vectors;

typedef struct {
  double time_sequential;
  double time_initialization;
  double time_reduction;
  double time_critical;
  double time_atomic;
} elapsed_time_stats;

double g_sum_sequential = 0.0;
double g_sum_reduction = 0.0;
double g_sum_critical = 0.0;
double g_sum_atomic = 0.0;

void print_initial_data() {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 3\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of threads: %d\n", NUM_OF_THREADS);
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");
}

void print_final_results(elapsed_time_stats *stats) {
    printf("====================== RESULTS ======================\n");
    printf("Sequential sum = %f\n", g_sum_sequential);
    printf("Reduction sum = %f\n", g_sum_reduction);
    printf("Critical sum = %f\n", g_sum_critical);
    printf("Atomic sum = %f\n\n", g_sum_atomic);

    printf("Sequential time = %f\n", stats->time_sequential);
    printf("Initialization time = %f\n", stats->time_initialization);
    printf("Reduction time = %f\n", stats->time_reduction);
    printf("Critical time = %f\n", stats->time_critical);
    printf("Atomic time = %f\n\n", stats->time_atomic);

    printf("With initialization:\n");
    printf("Reduction acceleration: %f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_reduction));
    printf("Critical acceleration: %f\n",
           stats->time_sequential / (stats->time_initialization + stats->time_critical));
    printf("Atomic acceleration: %f\n\n",
           stats->time_sequential / (stats->time_initialization + stats->time_atomic));

    printf("Without initialization:\n");
    printf("Reduction acceleration: %f\n", stats->time_sequential / stats->time_reduction);
    printf("Critical acceleration: %f\n", stats->time_sequential / stats->time_critical);
    printf("Atomic acceleration: %f\n", stats->time_sequential / stats->time_atomic);
    printf("=====================================================\n");
}

void sum_sequential_for(vectors *v) {
    g_sum_sequential = 0.0;
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            g_sum_sequential += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_reduction_for(vectors *v) {
    #pragma omp for collapse(2) reduction(+:g_sum_reduction)
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            g_sum_reduction += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
        }
    }
}

void sum_critical_for(vectors *v) {
    #pragma omp for collapse(2)
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            double val = v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
            #pragma omp critical
            {
                g_sum_critical += val;
            }
        }
    }
}

void sum_atomic_for(vectors *v) {
    #pragma omp for collapse(2)
    for (int j = 0; j < NMAX; ++j) {
        for (int q = 0; q < Q; ++q) {
            double val = v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
            #pragma omp atomic
            g_sum_atomic += val;
        }
    }
}

void normalize_results(elapsed_time_stats *stats) {
    g_sum_sequential /= Q;
    g_sum_reduction /= Q;
    g_sum_critical /= Q;
    g_sum_atomic /= Q;

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
    v.vec_1 = malloc(sizeof(double) * NMAX);
    v.vec_2 = malloc(sizeof(double) * NMAX);
    v.vec_3 = malloc(sizeof(double) * NMAX);

    for (int j = 0; j < NMAX; ++j) {
        v.vec_1[j] = 1.0;
        v.vec_2[j] = 1.0;
        v.vec_3[j] = 1.0;
    }

    elapsed_time_stats stats;
    double start_time;

    memset(&stats, 0, sizeof(stats));

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        g_sum_sequential = 0.0;
        g_sum_reduction = 0.0;
        g_sum_critical = 0.0;
        g_sum_atomic = 0.0;

        start_time = omp_get_wtime();
        sum_sequential_for(&v);
        stats.time_sequential += omp_get_wtime() - start_time;

        start_time = omp_get_wtime();

        #pragma omp parallel shared(v, stats, start_time)
        {
            #pragma omp barrier
            #pragma omp single
            {
                stats.time_initialization += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_reduction_for(&v);

            #pragma omp single
            {
                stats.time_reduction += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_critical_for(&v);

            #pragma omp single
            {
                stats.time_critical += omp_get_wtime() - start_time;
                start_time = omp_get_wtime();
            }

            sum_atomic_for(&v);

            #pragma omp single
            {
                stats.time_atomic += omp_get_wtime() - start_time;
            }
        }
    }

    normalize_results(&stats);
    print_final_results(&stats);

    free(v.vec_1);
    free(v.vec_2);
    free(v.vec_3);

    return 0;
}
