#include <omp.h>
#include <stdio.h>

#define NUM_OF_THREADS 16

int main(void) {
	omp_set_num_threads(NUM_OF_THREADS);
	int num_of_threads, thread_num;
	
	double start_time = omp_get_wtime();
	
	#pragma omp parallel private(num_of_threads, thread_num)
	{
		num_of_threads = omp_get_num_threads(); 
		thread_num = omp_get_thread_num(); 
		printf("OpenMP thread %d from %d threads \n", thread_num, num_of_threads);
	}

	double end_time = omp_get_wtime();

	printf("Execution time: %f seconds\n", end_time - start_time);

	return 0;
}
