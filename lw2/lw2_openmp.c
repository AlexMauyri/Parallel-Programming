#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_OF_VECTORS 3
#define NMAX 100000
#define NUM_OF_THREADS 2
#define ITER_NUM 20
#define Q 20

int main(int argc, char* argv[])
{
  printf("====================== PARAMS =======================\n");
  printf("Number of vectors: %d\n", NUM_OF_VECTORS);
  printf("Number of elements: %d\n", NMAX);
  printf("Number of threads: %d\n", NUM_OF_THREADS);
  printf("Number of iterations: %d\n", ITER_NUM);
  printf("Q: %d\n", Q);
  printf("=====================================================\n");

	omp_set_num_threads(NUM_OF_THREADS);
	
	double **a;
	a = malloc(sizeof(*a) * NUM_OF_VECTORS);
  for (int i = 0; i < NUM_OF_VECTORS; ++i) {
    a[i] = malloc(sizeof(**a) * NMAX);
	  for (int j = 0; j < NMAX; ++j) {
	    a[i][j] = 1.0;
	  }
  }

  double time_init = 0.0, time_sequential = 0.0, time_reduction = 0.0, time_critical = 0.0, time_atomic = 0.0, start_time;
  double sum_sequential, sum_reduction, sum_critical, sum_atomic;

  for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
    sum_sequential = sum_reduction = sum_critical = sum_atomic = 0.0;
	  start_time = omp_get_wtime();

	  for (int i = 0; i < NUM_OF_VECTORS; ++i) {
	    for (int j = 0; j < NMAX; ++j) {
	      for (int q = 0; q < Q; ++q) {
	        sum_sequential += a[i][j];
        }
	    }
	  }

	  time_sequential += omp_get_wtime() - start_time;
    
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
      #pragma omp barrier
      #pragma omp single 
      {
        time_init += omp_get_wtime() - start_time;
	      start_time = omp_get_wtime();
	    }

      #pragma omp for collapse(3) reduction(+:sum_reduction)
	    for (int i = 0; i < NUM_OF_VECTORS; ++i) {
	      for (int j = 0; j < NMAX; ++j) {
	        for (int q = 0; q < Q; ++q) {
	          sum_reduction += a[i][j];
	        }
	      }
	    }
    
      #pragma omp single 
      {
	      time_reduction += omp_get_wtime() - start_time;
	      start_time = omp_get_wtime();
      }

      #pragma omp for collapse(3)
	    for (int i = 0; i < NUM_OF_VECTORS; ++i) {
	      for (int j = 0; j < NMAX; ++j) {
	        for (int q = 0; q < Q; ++q) {
	          #pragma omp critical 
	          {
	            sum_critical += a[i][j];
	          }
	        }
	      }
	    }

      #pragma omp single 
      {
	      time_critical += omp_get_wtime() - start_time;
	      start_time = omp_get_wtime();
      }

      #pragma omp for collapse(3)
	    for (int i = 0; i < NUM_OF_VECTORS; ++i) {
	      for (int j = 0; j < NMAX; ++j) {
	        for (int q = 0; q < Q; ++q) {
	          #pragma omp atomic 
	          sum_atomic += a[i][j];
	        }
	      }
	    }

      #pragma omp single
	    time_atomic += omp_get_wtime() - start_time;
    }
  }

  printf("====================== RESULTS ======================\n");
  printf("Sequential sum = %f\n", sum_sequential / Q);
  printf("Reduction sum = %f\n", sum_reduction / Q);
  printf("Critical sum = %f\n", sum_critical / Q);
  printf("Atomic sum = %f\n\n", sum_atomic / Q);
  
  printf("Sequential time = %f\n", time_sequential / ITER_NUM);
  printf("Initialization time = %f\n", time_init / ITER_NUM);
  printf("Reduction time = %f\n", time_reduction / ITER_NUM);
  printf("Critical time = %f\n", time_critical / ITER_NUM);
  printf("Atomic time = %f\n\n", time_atomic / ITER_NUM);

  printf("With initialization:\n");
  printf("Reduction acceleration: %f\n", time_sequential / (time_init + time_reduction));
  printf("Critical acceleration: %f\n", time_sequential / (time_init + time_critical));
  printf("Atomic acceleration: %f\n\n", time_sequential / (time_init + time_atomic));

  printf("Without initialization:\n");
  printf("Reduction acceleration: %f\n", time_sequential / time_reduction);
  printf("Critical acceleration: %f\n", time_sequential / time_critical);
  printf("Atomic acceleration: %f\n", time_sequential / time_atomic);
  printf("=====================================================\n");
	return 0;
}

