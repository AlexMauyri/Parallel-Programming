#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX            1200000
#define NUM_OF_THREADS  10
#define ITER_NUM        20
#define Q               35
#define CHUNK           10000

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

int main(void) {
    omp_set_num_threads(NUM_OF_THREADS);

    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 2\n");
    printf("Vector type: int\n");
    printf("Number of elements: %d\n", NMAX);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Number of iterations: %d\n", ITER_NUM);
    printf("Q: %d\n", Q);
    printf("=====================================================\n");

    size_t vector_size = NMAX;
    int *vec_1, *vec_2;

    vec_1 = malloc(sizeof(int) * vector_size);
    vec_2 = malloc(sizeof(int) * vector_size);

    if (!vec_1 || !vec_2) {
        fprintf(stderr, "Error: memory not allocated for vectors!\n");
        return -1;
    }

    size_t sums_size = NMAX;
    int *sum_sequential, *sum_static, *sum_dynamic, *sum_guided;

    sum_sequential = malloc(sizeof(int) * sums_size);
    sum_static = malloc(sizeof(int) * sums_size);
    sum_dynamic = malloc(sizeof(int) * sums_size);
    sum_guided = malloc(sizeof(int) * sums_size);

    if (!sum_sequential || !sum_static || !sum_dynamic || !sum_guided) {
        fprintf(stderr, "Error: memory not allocated for sum vectors!\n");
        return -1;
    }

    for (size_t i = 0; i < vector_size; ++i) {
        vec_1[i] = 1;
        vec_2[i] = 1;
    }

    double time_sequential = 0.0, time_initialization = 0.0;
    double time_static = 0.0, time_dynamic = 0.0, time_guided = 0.0;
    double start_time;

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        memset(sum_sequential, 0, sums_size * sizeof(int));
        memset(sum_static, 0, sums_size * sizeof(int));
        memset(sum_dynamic, 0, sums_size * sizeof(int));
        memset(sum_guided, 0, sums_size * sizeof(int));

        start_time = omp_get_wtime();
        for (size_t i = 0; i < vector_size; ++i) {
            for (int q = 0; q < Q; ++q) {
                sum_sequential[i] += vec_1[i] + vec_2[i];
            }
        }
        time_sequential += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);

        start_time = omp_get_wtime();

        #pragma omp parallel default(shared)
        {
            #pragma omp barrier
            #pragma omp single
            {
                time_initialization += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            #pragma omp for schedule(static, CHUNK)
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_static[i] += vec_1[i] + vec_2[i];
                }
            }

            #pragma omp single
            {
                time_static += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            #pragma omp for schedule(dynamic, CHUNK)
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_dynamic[i] += vec_1[i] + vec_2[i];
                }
            }

            #pragma omp single
            {
                time_dynamic += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            #pragma omp for schedule(guided, CHUNK)
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_guided[i] += vec_1[i] + vec_2[i];
                }
            }

            #pragma omp single
            {
                time_guided += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
            }
        }
    }

    for (size_t i = 0; i < sums_size; ++i) {
        sum_sequential[i] /= Q;
        sum_static[i] /= Q;
        sum_dynamic[i] /= Q;
        sum_guided[i] /= Q;
    }

    time_sequential /= ITER_NUM;
    time_initialization /= ITER_NUM;
    time_static /= ITER_NUM;
    time_dynamic /= ITER_NUM;
    time_guided /= ITER_NUM;

    int static_equal = 1;
    for (size_t i = 0; i < sums_size; ++i) {
        if (sum_sequential[i] != sum_static[i]) {
            static_equal = 0;
            break;
        }
    }

    int dynamic_equal = 1;
    for (size_t i = 0; i < sums_size; ++i) {
        if (sum_sequential[i] != sum_dynamic[i]) {
            dynamic_equal = 0;
            break;
        }
    }

    int guided_equal = 1;
    for (size_t i = 0; i < sums_size; ++i) {
        if (sum_sequential[i] != sum_guided[i]) {
            guided_equal = 0;
            break;
        }
    }

    printf("====================== RESULTS ======================\n");
    printf("Static sum is equal to sequential sum: %s\n", static_equal ? "Yes" : "No");
    printf("Dynamic sum is equal to sequential sum: %s\n", dynamic_equal ? "Yes" : "No");
    printf("Guided sum is equal to sequential sum: %s\n\n", guided_equal ? "Yes" : "No");

    printf("Sequential time = %.4f ms\n", time_sequential);
    printf("Initialization time = %.4f ms\n", time_initialization);
    printf("Static time = %.4f ms\n", time_static);
    printf("Dynamic time = %.4f ms\n", time_dynamic);
    printf("Guided time = %.4f ms\n\n", time_guided);

    printf("With initialization:\n");
    printf("Static acceleration: %.4f\n", time_sequential / (time_initialization + time_static));
    printf("Dynamic acceleration: %.4f\n", time_sequential / (time_initialization + time_dynamic));
    printf("Guided acceleration: %.4f\n\n", time_sequential / (time_initialization + time_guided));

    printf("Without initialization:\n");
    printf("Static acceleration: %.4f\n", time_sequential / time_static);
    printf("Dynamic acceleration: %.4f\n", time_sequential / time_dynamic);
    printf("Guided acceleration: %.4f\n", time_sequential / time_guided);
    printf("=====================================================\n");

    free(vec_1);
    free(vec_2);
    free(sum_sequential);
    free(sum_static);
    free(sum_dynamic);
    free(sum_guided);

    return 0;
}