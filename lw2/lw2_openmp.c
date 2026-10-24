#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX            12000000
#define ITER_NUM        20
#define Q               35

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

int main(void) {
    printf("====================== PARAMS =======================\n");
    printf("Number of vectors: 2\n");
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
        fprintf(stderr, "Error: memory not allocated for vectors\n");
        return -1;
    }

    for (size_t i = 0; i < vector_size; ++i) {
        vec_1[i] = 1;
        vec_2[i] = 1;
    }

    int sum_sequential = 0, sum_reduction = 0, sum_critical = 0, sum_atomic = 0;
    double time_sequential = 0.0, time_initialization = 0.0, time_reduction = 0.0;
    double time_critical = 0.0, time_atomic = 0.0;
    double start_time;

    for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
        sum_sequential = 0;
        sum_reduction = 0;
        sum_critical = 0;
        sum_atomic = 0;

        start_time = omp_get_wtime();
        for (size_t i = 0; i < vector_size; ++i) {
            for (int q = 0; q < Q; ++q) {
                sum_sequential += vec_1[i] + vec_2[i];
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

            #pragma omp for reduction(+:sum_reduction)
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    sum_reduction += vec_1[i] + vec_2[i];
                }
            }

            #pragma omp single
            {
                time_reduction += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            #pragma omp for
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    #pragma omp critical
                    {
                        sum_critical += vec_1[i] + vec_2[i];
                    }
                }
            }

            #pragma omp single
            {
                time_critical += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
                start_time = omp_get_wtime();
            }

            #pragma omp for
            for (size_t i = 0; i < vector_size; ++i) {
                for (int q = 0; q < Q; ++q) {
                    #pragma omp atomic
                    sum_atomic += vec_1[i] + vec_2[i];
                }
            }

            #pragma omp single
            {
                time_atomic += SECONDS_TO_MILLISECONDS(omp_get_wtime() - start_time);
            }
        }
    }

    sum_sequential /= Q;
    sum_reduction /= Q;
    sum_critical /= Q;
    sum_atomic /= Q;

    time_sequential /= ITER_NUM;
    time_initialization /= ITER_NUM;
    time_reduction /= ITER_NUM;
    time_critical /= ITER_NUM;
    time_atomic /= ITER_NUM;

    printf("====================== RESULTS ======================\n");
    printf("Sequential sum = %d\n", sum_sequential);
    printf("Reduction sum = %d\n", sum_reduction);
    printf("Critical sum = %d\n", sum_critical);
    printf("Atomic sum = %d\n\n", sum_atomic);

    printf("Sequential time = %.4f ms\n", time_sequential);
    printf("Initialization time = %.4f ms\n", time_initialization);
    printf("Reduction time = %.4f ms\n", time_reduction);
    printf("Critical time = %.4f ms\n", time_critical);
    printf("Atomic time = %.4f ms\n\n", time_atomic);

    printf("With initialization:\n");
    printf("Reduction acceleration: %.4f\n", time_sequential / (time_initialization + time_reduction));
    printf("Critical acceleration: %.4f\n", time_sequential / (time_initialization + time_critical));
    printf("Atomic acceleration: %.4f\n\n", time_sequential / (time_initialization + time_atomic));

    printf("Without initialization:\n");
    printf("Reduction acceleration: %.4f\n", time_sequential / time_reduction);
    printf("Critical acceleration: %.4f\n", time_sequential / time_critical);
    printf("Atomic acceleration: %.4f\n", time_sequential / time_atomic);
    printf("=====================================================\n");

    free(vec_1);
    free(vec_2);

    return 0;
}