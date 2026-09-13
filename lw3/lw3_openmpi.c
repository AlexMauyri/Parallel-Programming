#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NMAX 1000000
#define ADDITION 133
#define NMAX_NON_MUL (NMAX + ADDITION)
#define ITER_NUM 20
#define Q 20
#define EPSILON 1e-9

#define SECONDS_TO_MILLISECONDS(seconds) ((seconds) * 1e3)

typedef struct {
	double *vec_1;
	double *vec_2;
	double *vec_3;
	size_t size;
} vectors;

typedef struct {
  double *sum_sequential;
	double *sum_parallel;
	size_t size;
} sum_vectors;

typedef struct {
  double time_sequential_mul;
  double time_sequential_non_mul;
  double time_scatter_cast;
  double time_scatterv_cast;
  double time_parallel_mul;
  double time_parallel_non_mul;
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

int vectors_malloc(vectors *v, const size_t size) {
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
  free(sums->sum_parallel);
  sums->sum_sequential = NULL;
  sums->sum_parallel = NULL;
  sums->size = 0;
}

int sum_vectors_malloc(sum_vectors *sums, const size_t size) {
  sums->sum_sequential = malloc(sizeof(double) * size);
  sums->sum_parallel = malloc(sizeof(double) * size);
  sums->size = size;

  if (!sums->sum_sequential || !sums->sum_parallel) {
    sum_vectors_free(sums);  
    return -1;
  }

  return 0;
}

int arrays_equal(const double *a, const double *b, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > EPSILON) {
            return 0;
        }
    }

    return 1;
}

void print_initial_data(void) {
  printf("====================== PARAMS =======================\n");
  printf("Number of vectors: 3\n");
  printf("Number of elements for mul: %d\n", NMAX);
  printf("Number of elements for non mul: %d\n", NMAX_NON_MUL);
  printf("Number of iterations: %d\n", ITER_NUM);
  printf("Q: %d\n", Q);
  printf("=====================================================\n");
}

void print_final_results(const sum_vectors *sums_mul, const sum_vectors *sums_non_mul, const elapsed_time_stats *stats) {
  printf("====================== RESULTS ======================\n");
  printf("Mul parallel sum is equal mul sequential sum: %s\n",
          arrays_equal(sums_mul->sum_sequential, sums_mul->sum_parallel, sums_mul->size) ? "Yes" : "No");
  printf("Non mul parallel sum is equal non mul sequential sum: %s\n\n",
          arrays_equal(sums_non_mul->sum_sequential, sums_non_mul->sum_parallel, sums_non_mul->size) ? "Yes" : "No");

  printf("Mul sequential time = %.4f ms\n", stats->time_sequential_mul);
  printf("Non mul sequential time = %.4f ms\n", stats->time_sequential_non_mul);
  printf("Scatter cast time = %.4f ms\n", stats->time_scatter_cast);
  printf("Scatterv time = %.4f ms\n", stats->time_scatterv_cast);
  printf("Mul parallel time = %.4f ms\n", stats->time_parallel_mul);
  printf("Non mul parallel time = %.4f ms\n\n", stats->time_parallel_non_mul);

  printf("With scatter:\n");
  printf("Mul parallel acceleration: %.4f\n",
          stats->time_sequential_mul / (stats->time_scatter_cast + stats->time_parallel_mul));
  printf("Non mul acceleration: %.4f\n\n",
          stats->time_sequential_non_mul / (stats->time_scatterv_cast + stats->time_parallel_non_mul));

  printf("Without scatter:\n");
  printf("Mul parallel acceleration: %.4f\n", stats->time_sequential_mul / stats->time_parallel_mul);
  printf("Non mul acceleration: %.4f\n", stats->time_sequential_non_mul / stats->time_parallel_non_mul);
  printf("=====================================================\n");
}

void vectors_sum(const vectors *v, double *sum_result) {
  for (size_t i = 0; i < v->size; ++i) {
    for (int q = 0; q < Q; ++q) {
      sum_result[i] += v->vec_1[i] + v->vec_2[i] + v->vec_3[i];
    }
  }
}

void normalize_results(sum_vectors *sums_mul, sum_vectors *sums_non_mul, elapsed_time_stats *stats) {
  for (size_t i = 0; i < sums_mul->size; ++i) {
      sums_mul->sum_sequential[i] /= Q;
      sums_mul->sum_parallel[i] /= Q;
  }
	for (size_t i = 0; i < sums_non_mul->size; ++i) {
      sums_non_mul->sum_sequential[i] /= Q;
      sums_non_mul->sum_parallel[i] /= Q;
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
 
 	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &process_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  const size_t vector_size_mul = NMAX;
  const size_t vector_size_non_mul = NMAX_NON_MUL;
  
  int block_size_mul = vector_size_mul / process_num;
  int block_size_non_mul = vector_size_non_mul / process_num + (process_rank < NMAX_NON_MUL % process_num ? 1 : 0);

  vectors v_mul = {NULL, NULL, NULL, 0};
  vectors v_non_mul = {NULL, NULL, NULL, 0};
  vectors v_buf_mul = {NULL, NULL, NULL, 0};
  vectors v_buf_non_mul = {NULL, NULL, NULL, 0};
  sum_vectors sums_mul = {NULL, NULL, 0};
  sum_vectors sums_non_mul = {NULL, NULL, 0};
  double *sum_parallel_mul_buf = NULL, *sum_parallel_non_mul_buf = NULL;
  int *counts = NULL, *displs = NULL;

	if (process_rank == 0) {  
		print_initial_data();
		
  	if (vectors_malloc(&v_mul, vector_size_mul)) {
  	  fprintf(stderr, "Error: memory not allocated for mul vectors!\n");
  	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  	}
    
    if (vectors_malloc(&v_non_mul, vector_size_non_mul)) {
  	  fprintf(stderr, "Error: memory not allocated for non mul vectors!\n");
  	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  	}
  
    if (sum_vectors_malloc(&sums_mul, vector_size_mul)) {
      fprintf(stderr, "Error: memory not allocated for mul sum vectors!\n");
  	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (sum_vectors_malloc(&sums_non_mul, vector_size_non_mul)) {
      fprintf(stderr, "Error: memory not allocated for non mul sum vectors!\n");
  	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

  	counts = malloc(sizeof(int) * process_num);
  	displs = malloc(sizeof(int) * process_num);

  	if (!counts || !displs) {
      fprintf(stderr, "Error: memory not allocated for counts/displs\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

  	for (size_t i = 0; i < v_mul.size; ++i) {
    	v_mul.vec_1[i] = 1.0;
    	v_mul.vec_2[i] = 1.0;
    	v_mul.vec_3[i] = 1.0;
  	}

  	for (size_t i = 0; i < v_non_mul.size; ++i) {
    	v_non_mul.vec_1[i] = 1.0;
    	v_non_mul.vec_2[i] = 1.0;
    	v_non_mul.vec_3[i] = 1.0;
  	}

		int accum = 0;
		int base = vector_size_non_mul / process_num;
		int rem = vector_size_non_mul % process_num;
  	for (int i = 0; i < process_num; ++i) {
  		counts[i] = base + (i < rem ? 1 : 0);
  		displs[i] = accum;
  		accum += counts[i];
  	}

  }

	if (vectors_malloc(&v_buf_mul, block_size_mul)) {
  	fprintf(stderr, "Error: memory not allocated for mul vectors buffer in process %d\n!", process_rank);
  	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (vectors_malloc(&v_buf_non_mul, block_size_non_mul)) {
  	fprintf(stderr, "Error: memory not allocated for non mul vectors buffer in process %d\n!", process_rank);
  	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  sum_parallel_mul_buf = malloc(sizeof(double) * block_size_mul);

  if (sum_parallel_mul_buf == NULL) {
    fprintf(stderr, "Error: memory not allocated for mul sum vector buffer in process %d\n!", process_rank);
  	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  sum_parallel_non_mul_buf = malloc(sizeof(double) * block_size_non_mul);
  
  if (sum_parallel_non_mul_buf == NULL) {
    fprintf(stderr, "Error: memory not allocated for non mul sum vector buffer in process %d\n!", process_rank);
  	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  
  elapsed_time_stats stats;
  double start_time;

  memset(&stats, 0, sizeof(stats));

	for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
		if (process_rank == 0) {
    	memset(sums_mul.sum_sequential, 0, sums_mul.size * sizeof(double));
    	memset(sums_non_mul.sum_sequential, 0, sums_non_mul.size * sizeof(double));
    	memset(sums_mul.sum_parallel, 0, sums_mul.size * sizeof(double));
    	memset(sums_non_mul.sum_parallel, 0, sums_non_mul.size * sizeof(double));
		}
		memset(sum_parallel_mul_buf, 0, block_size_mul * sizeof(double));
    memset(sum_parallel_non_mul_buf, 0, block_size_non_mul * sizeof(double));

    if (process_rank == 0) {
      start_time = MPI_Wtime();
    	vectors_sum(&v_mul, sums_mul.sum_sequential);
    	stats.time_sequential_mul += MPI_Wtime() - start_time;

    	start_time = MPI_Wtime();
    	vectors_sum(&v_non_mul, sums_non_mul.sum_sequential);
    	stats.time_sequential_non_mul += MPI_Wtime() - start_time;
    }
	
    MPI_Barrier(MPI_COMM_WORLD);

		start_time = MPI_Wtime();
		MPI_Scatter(v_mul.vec_1, block_size_mul, MPI_DOUBLE, v_buf_mul.vec_1, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(v_mul.vec_2, block_size_mul, MPI_DOUBLE, v_buf_mul.vec_2, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(v_mul.vec_3, block_size_mul, MPI_DOUBLE, v_buf_mul.vec_3, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_scatter_cast += MPI_Wtime() - start_time;

    MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		vectors_sum(&v_buf_mul, sum_parallel_mul_buf);
		MPI_Gather(sum_parallel_mul_buf, block_size_mul, MPI_DOUBLE, sums_mul.sum_parallel, block_size_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_parallel_mul += MPI_Wtime() - start_time;
		
    MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		MPI_Scatterv(v_non_mul.vec_1, counts, displs, MPI_DOUBLE, v_buf_non_mul.vec_1, block_size_non_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(v_non_mul.vec_2, counts, displs, MPI_DOUBLE, v_buf_non_mul.vec_2, block_size_non_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(v_non_mul.vec_3, counts, displs, MPI_DOUBLE, v_buf_non_mul.vec_3, block_size_non_mul, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_scatterv_cast += MPI_Wtime() - start_time;

    MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		vectors_sum(&v_buf_non_mul, sum_parallel_non_mul_buf);
		MPI_Gatherv(sum_parallel_non_mul_buf, block_size_non_mul, MPI_DOUBLE, sums_non_mul.sum_parallel, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		stats.time_parallel_non_mul += MPI_Wtime() - start_time;
	}

	if (process_rank == 0) {
  	normalize_results(&sums_mul, &sums_non_mul, &stats);
  	print_final_results(&sums_mul, &sums_non_mul, &stats);

    vectors_free(&v_mul);	
    vectors_free(&v_non_mul);
    sum_vectors_free(&sums_mul);
    sum_vectors_free(&sums_non_mul);    
    free(counts);
    free(displs);
  }

  vectors_free(&v_buf_mul);
  vectors_free(&v_buf_non_mul);
  free(sum_parallel_mul_buf);
  free(sum_parallel_non_mul_buf);
  MPI_Finalize();
	return 0;
}