#include <stdio.h>
#include <mpi.h>

#define ITER_NUM 12

int main(int argc, char* argv[]) {
	int rank, rank_size;
	double total_time = 0.0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	
	for (int iteration = 0; iteration < ITER_NUM; ++iteration) {	
		MPI_Barrier(MPI_COMM_WORLD);
		
		const double start_time = MPI_Wtime();
			
		printf("Hello world from process %d from total number of %d\n", rank, rank_size);

		MPI_Barrier(MPI_COMM_WORLD);
		
		const double end_time = MPI_Wtime();
		total_time += end_time - start_time;
	}

	MPI_Finalize();

	if (rank == 0) {
		printf("Average time: %f seconds on %d iterations\n", total_time / ITER_NUM, ITER_NUM);
	}
	return 0;
}
