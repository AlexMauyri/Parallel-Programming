#include <omp.h>
#include <stdio.h>

#define NUM_OF_THREADS 16

int main(int argc, char* argv[]) {
	omp_set_num_threads(NUM_OF_THREADS);
	int nThreads, threadNum;
	
	double start_time = omp_get_wtime();
	
	#pragma omp parallel private(nThreads, threadNum)
	{
		nThreads = omp_get_num_threads(); 
		threadNum = omp_get_thread_num(); 
		printf("OpenMP thread %d from %d threads \n", threadNum, nThreads);
	}

	double end_time = omp_get_wtime();

	printf("Execution time: %f seconds\n", end_time - start_time);

	return 0;
}
