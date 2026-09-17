#include <omp.h>
#include <stdio.h>

#define NUM_OF_THREADS 	4
#define ITER_NUM 				12

int main(void) {
	omp_set_num_threads(NUM_OF_THREADS);
	int num_of_threads, thread_num;
	double total_time = 0.0;

	for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
		const double start_time = omp_get_wtime();
		
		#pragma omp parallel private(num_of_threads, thread_num)
		{
			num_of_threads = omp_get_num_threads(); 
			thread_num = omp_get_thread_num(); 
			printf("OpenMP thread %d from %d threads \n", thread_num, num_of_threads);
		}

		const double end_time = omp_get_wtime();

		total_time += end_time - start_time;
	}

	printf("Average time: %f seconds on %d iterations\n", total_time / ITER_NUM, ITER_NUM);
	return 0;
}
