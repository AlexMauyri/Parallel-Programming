#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define NMAX 100000
#define ADDITION 13333
#define NMAX_NON_MUL (NMAX + ADDITION)
#define ITER_NUM 20
#define Q 20

typedef struct {
	double *vec_1;
	double *vec_2;
	double *vec_3;
} vectors;

typedef struct {
  double *sum_sequential_mul;
  double *sum_sequential_non_mul;
	double *sum_parallel_mul;
	double *sum_parallel_non_mul;
} sum_arrays;

typedef struct {
  double time_sequential_mul;
  double time_sequential_non_mul;
  double time_scatter_cast;
  double time_scatterv_cast;
  double time_parallel_mul;
  double time_parallel_non_mul;
} elapsed_time_stats;

void print_initial_data() {
  printf("====================== PARAMS =======================\n");
  printf("Number of vectors: 3\n");
  printf("Number of elements: %d\n", NMAX);
  printf("Number of iterations: %d\n", ITER_NUM);
  printf("Q: %d\n", Q);
  printf("=====================================================\n");
}

void print_final_results(sum_arrays *sums, elapsed_time_stats *stats) {
  printf("====================== RESULTS ======================\n");
  printf("Mul parallel sum is equal mul sequential sum: %s\n",
          memcmp(sums->sum_sequential_mul, sums->sum_parallel_mul, NMAX * sizeof(double)) ? "No" : "Yes");
  printf("Non mul parallel sum is equal non mul sequential sum: %s\n",
          memcmp(sums->sum_sequential_non_mul, sums->sum_parallel_non_mul, NMAX * sizeof(double)) ? "No" : "Yes");

  printf("Mul sequential time = %f\n", stats->time_sequential_mul);
  printf("Non mul sequential time = %f\n", stats->time_sequential_non_mul);
  printf("Scatter cast time = %f\n", stats->time_scatter_cast);
  printf("Scatterv time = %f\n", stats->time_scatterv_cast);
  printf("Mul parallel time = %f\n", stats->time_parallel_mul);
  printf("Non mul parallel time = %f\n\n", stats->time_parallel_non_mul);

  printf("With scatter:\n");
  printf("Mul parallel acceleration: %f\n",
          stats->time_sequential_mul / (stats->time_scatter_cast + stats->time_parallel_mul));
  printf("Non mul acceleration: %f\n",
          stats->time_sequential_non_mul / (stats->time_scatterv_cast + stats->time_parallel_non_mul));

  printf("Without scatter:\n");
  printf("Mul parallel acceleration: %f\n", stats->time_sequential_mul / stats->time_parallel_mul);
  printf("Non mul acceleration: %f\n", stats->time_sequential_non_mul / stats->time_parallel_non_mul);
  printf("=====================================================\n");
}

void sum_vectors(vectors *v, double *sum_result, int size) {
  for (int j = 0; j < size; ++j) {
    for (int q = 0; q < Q; ++q) {
      sum_result[j] += v->vec_1[j] + v->vec_2[j] + v->vec_3[j];
    }
  }
}

void normalize_results(sum_arrays *sums, elapsed_time_stats *stats) {
    for (int j = 0; j < NMAX; ++j) {
        sums->sum_sequential_mul[j] /= Q;
        sums->sum_parallel_mul[j] /= Q;
    }
		for (int j = 0; j < NMAX_NON_MUL; ++j) {
        sums->sum_sequential_non_mul[j] /= Q;
        sums->sum_parallel_non_mul[j] /= Q;
    }

    stats->time_sequential_mul /= ITER_NUM;
    stats->time_sequential_non_mul /= ITER_NUM;
    stats->time_scatter_cast /= ITER_NUM;
    stats->time_scatterv_cast /= ITER_NUM;
    stats->time_parallel_mul /= ITER_NUM;
    stats->time_parallel_non_mul /= ITER_NUM;
}

int main(int argc, char* argv[]) {
  int process_rank, process_num;
	MPI_Status status;
 
 	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &process_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  
  int block_size_mul = NMAX / process_num, block_size_non_mul = NMAX_NON_MUL / process_num + (process_rank < NMAX_NON_MUL % process_num ? 1 : 0);

  vectors v_mul, v_non_mul, v_buf_mul, v_buf_non_mul;
  sum_arrays sums, sums_buf;
  int *sendcounts, *displs;


	
	if (process_rank == 0) {  
		print_initial_data();
		
		v_mul.vec_1 = malloc(sizeof(double) * NMAX);
  	v_mul.vec_2 = malloc(sizeof(double) * NMAX);
  	v_mul.vec_3 = malloc(sizeof(double) * NMAX);

  	v_non_mul.vec_1 = malloc(sizeof(double) * NMAX_NON_MUL);
  	v_non_mul.vec_2 = malloc(sizeof(double) * NMAX_NON_MUL);
  	v_non_mul.vec_3 = malloc(sizeof(double) * NMAX_NON_MUL);

  	sums.sum_sequential_mul = malloc(sizeof(double) * NMAX);
  	sums.sum_sequential_non_mul = malloc(sizeof(double) * NMAX_NON_MUL);
  	sums.sum_parallel_mul = malloc(sizeof(double) * NMAX);
  	sums.sum_parallel_non_mul = malloc(sizeof(double) * NMAX_NON_MUL);

  	sendcounts = malloc(sizeof(int) * process_num);
  	displs = malloc(sizeof(int) * process_num);

  	for (int j = 0; j < NMAX; ++j) {
    	v_mul.vec_1[j] = 1.0;
    	v_mul.vec_2[j] = 1.0;
    	v_mul.vec_3[j] = 1.0;
  	}

  	for (int j = 0; j < NMAX_NON_MUL; ++j) {
    	v_non_mul.vec_1[j] = 1.0;
    	v_non_mul.vec_2[j] = 1.0;
    	v_non_mul.vec_3[j] = 1.0;
  	}

		int accum = 0;
  	for (int i = 0; i < process_num; ++i) {
  		sendcounts[i] = block_size_non_mul + (i < NMAX_NON_MUL % process_num ? 0 : -1);
  		displs[i] = accum;
  		accum += sendcounts[i];
  	}

  }

	v_buf_mul.vec_1 = malloc(sizeof(double) * block_size_mul);
  v_buf_mul.vec_2 = malloc(sizeof(double) * block_size_mul);
  v_buf_mul.vec_3 = malloc(sizeof(double) * block_size_mul);

  v_buf_non_mul.vec_1 = malloc(sizeof(double) * block_size_non_mul);
  v_buf_non_mul.vec_2 = malloc(sizeof(double) * block_size_non_mul);
  v_buf_non_mul.vec_3 = malloc(sizeof(double) * block_size_non_mul);
  
  sums_buf.sum_parallel_mul = malloc(sizeof(double) * block_size_mul);
  sums_buf.sum_parallel_non_mul = malloc(sizeof(double) * block_size_non_mul);

  elapsed_time_stats stats;
  double start_time;

  memset(&stats, 0, sizeof(stats));

	for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
		if (process_rank == 0) {
    	memset(sums.sum_sequential_mul, 0, NMAX * sizeof(double));
    	memset(sums.sum_sequential_non_mul, 0, NMAX_NON_MUL * sizeof(double));
    	memset(sums.sum_parallel_mul, 0, NMAX * sizeof(double));
    	memset(sums.sum_parallel_non_mul, 0, NMAX_NON_MUL * sizeof(double));
    
    	start_time = MPI_Wtime();
    	sum_vectors(&v_mul, sums.sum_sequential_mul, NMAX);
    	stats.time_sequential_mul += MPI_Wtime() - start_time;

    	start_time = MPI_Wtime();
    	sum_vectors(&v_non_mul, sums.sum_sequential_non_mul, NMAX_NON_MUL);
    	stats.time_sequential_non_mul += MPI_Wtime() - start_time;
		}
		memset(sums_buf.sum_parallel_mul, 0, block_size_mul * sizeof(double));
    memset(sums_buf.sum_parallel_non_mul, 0, block_size_non_mul * sizeof(double));
		
		start_time = MPI_Wtime();
		MPI_Scatter(v_mul.vec_1, block_size_mul, MPI_DOUBLE, v_buf_mul.vec_1, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(v_mul.vec_2, block_size_mul, MPI_DOUBLE, v_buf_mul.vec_2, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(v_mul.vec_3, block_size_mul, MPI_DOUBLE, v_buf_mul.vec_3, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_scatter_cast = MPI_Wtime() - start_time;

		start_time = MPI_Wtime();
		sum_vectors(&v_buf_mul, sums_buf.sum_parallel_mul, block_size_mul);
		MPI_Gather(sums_buf.sum_parallel_mul, block_size_mul, MPI_DOUBLE, sums.sum_parallel_mul, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_parallel_mul = MPI_Wtime() - start_time;
		
		start_time = MPI_Wtime();
		MPI_Scatterv(v_non_mul.vec_1, sendcounts, displs, MPI_DOUBLE, v_buf_non_mul.vec_1, block_size_non_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(v_non_mul.vec_2, sendcounts, displs, MPI_DOUBLE, v_buf_non_mul.vec_2, block_size_non_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(v_non_mul.vec_3, sendcounts, displs, MPI_DOUBLE, v_buf_non_mul.vec_3, block_size_non_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_scatterv_cast = MPI_Wtime() - start_time;

		start_time = MPI_Wtime();
		sum_vectors(&v_buf_non_mul, sums_buf.sum_parallel_non_mul, block_size_non_mul);
		MPI_Gatherv(sums_buf.sum_parallel_non_mul, block_size_non_mul, MPI_DOUBLE, sums.sum_parallel_non_mul, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_parallel_non_mul = MPI_Wtime() - start_time;
	}

	if (process_rank == 0) {
  	normalize_results(&sums, &stats);
  	print_final_results(&sums, &stats);
	}

	MPI_Finalize();
	
	if (process_rank == 0) {
  	free(v_mul.vec_1);
  	free(v_mul.vec_2);
  	free(v_mul.vec_3);
  	free(v_non_mul.vec_1);
  	free(v_non_mul.vec_2);
  	free(v_non_mul.vec_3);

  	free(sums.sum_sequential_mul);
  	free(sums.sum_sequential_non_mul);
  	free(sums.sum_parallel_mul);
  	free(sums.sum_parallel_non_mul);
  }
  
  free(v_buf_mul.vec_1);
  free(v_buf_mul.vec_2);
  free(v_buf_mul.vec_3);
  free(v_buf_non_mul.vec_1);
  free(v_buf_non_mul.vec_2);
  free(v_buf_non_mul.vec_3); 
  

  free(sums_buf.sum_parallel_mul);
  free(sums_buf.sum_parallel_non_mul);
	return 0;
}
