#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <locale.h>
#include <stdio.h>


int main(int argc, char* argv[])
{
	omp_set_num_threads(16);
	int nThreads, threadNum;
	#pragma omp parallel  private(nThreads, threadNum)
	{
		nThreads = omp_get_num_threads(); 
		threadNum = omp_get_thread_num(); 
		printf("OpenMP thread ?%d from %d threads \n", threadNum, nThreads);
	}
	return 0;
}
