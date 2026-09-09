#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define NUM_OF_VECTORS 3
#define NMAX 100000
#define ITER_NUM 20
#define Q 20

int main(int argc, char* argv[]) {
  double **a;
	a = malloc(sizeof(*a) * NUM_OF_VECTORS);
  for (int i = 0; i < NUM_OF_VECTORS; ++i) {
    a[i] = malloc(sizeof(**a) * NMAX);
  }

	double total_sum_sequential = 0.0, total_sum_pp, total_sum_reduce, process_sum = 0.0;
	int process_rank, process_num;
	MPI_Status status;
	double time_sequential = 0.0, time_broadcast = 0.0, time_pp = 0.0, time_reduce = 0.0, start_time;
  
  MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &process_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  
	if (process_rank == 0) {
  	printf("====================== PARAMS =======================\n");
  	printf("Number of vectors: %d\n", NUM_OF_VECTORS);
  	printf("Number of elements: %d\n", NMAX);
  	printf("Number of iterations: %d\n", ITER_NUM);
  	printf("Q: %d\n", Q);
  	printf("=====================================================\n");
	}

  if (process_rank == 0) {
		for (int i = 0; i < NUM_OF_VECTORS; ++i) {
	  	for (int j = 0; j < NMAX; ++j) {
	    	a[i][j] = 1.0;
	  	}
  	}
	}

	int k = NMAX / process_num;
	int j1 = k * process_rank;
	int j2 = k * ( process_rank + 1 );
	
	if (process_rank == process_num - 1) {
		j2 = NMAX;
	}
	
	for (int iteration = 0; iteration < ITER_NUM; ++iteration) {
		total_sum_sequential = total_sum_pp = total_sum_reduce = 0.0;
	
		if (process_rank == 0) {
			start_time = MPI_Wtime();
			for (int i = 0; i < NUM_OF_VECTORS; ++i) {
				for (int j = 0; j < NMAX; ++j) {
					for (int q = 0; q < Q; ++q) {
						total_sum_sequential += a[i][j];
					}
				}
			}
			time_sequential += MPI_Wtime() - start_time;
		}
		
		start_time = MPI_Wtime();
		for (int i = 0; i < NUM_OF_VECTORS; ++i) {
			MPI_Bcast(a[i], NMAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		time_broadcast += MPI_Wtime() - start_time;

		MPI_Barrier(MPI_COMM_WORLD);

		process_sum = 0.0;
		start_time = MPI_Wtime();
		for (int i = 0; i < NUM_OF_VECTORS; ++i) {
			for (int j = j1; j < j2; ++j) {
				for (int q = 0; q < Q; ++q) {
					process_sum += a[i][j];
				}
			}
		}

		if ( process_rank == 0 ) {
			total_sum_pp = process_sum;
			for (int i = 1; i < process_num; ++i) {
				MPI_Recv(&process_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
				total_sum_pp += process_sum;
			}
		}
		else {
			MPI_Send(&process_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		time_pp += MPI_Wtime() - start_time;

		MPI_Barrier(MPI_COMM_WORLD);
		
		process_sum = 0.0;
		start_time = MPI_Wtime();
		for (int i = 0; i < NUM_OF_VECTORS; ++i) {
			for (int j = j1; j < j2; ++j) {
				for (int q = 0; q < Q; ++q) {
					process_sum += a[i][j];
				}
			}
		}

		MPI_Reduce(&process_sum, &total_sum_reduce, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		time_reduce += MPI_Wtime() - start_time;

		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	if (process_rank == 0) {
		printf("====================== RESULTS ======================\n");
  	printf("Sequential sum = %f\n", total_sum_sequential / Q);
  	printf("P2P sum = %f\n", total_sum_pp / Q);
  	printf("Reduce sum = %f\n", total_sum_reduce / Q);
  	
  	printf("Sequential time = %f\n", time_sequential / ITER_NUM);
  	printf("Broadcast time = %f\n", time_broadcast / ITER_NUM);
  	printf("P2P time = %f\n", time_pp / ITER_NUM);
  	printf("Reduce time = %f\n", time_reduce / ITER_NUM);

  	printf("With broadcast:\n");
  	printf("P2P acceleration: %f\n", time_sequential / (time_broadcast + time_pp));
  	printf("Reduce acceleration: %f\n", time_sequential / (time_broadcast + time_reduce));

  	printf("Without broadcast:\n");
  	printf("P2P acceleration: %f\n", time_sequential / time_pp);
  	printf("Reduce acceleration: %f\n", time_sequential / time_reduce);
  	printf("=====================================================\n");
	}

	MPI_Finalize();
	return 0;
}

