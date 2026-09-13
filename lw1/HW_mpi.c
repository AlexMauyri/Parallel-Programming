#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
	int rank, rank_size;
	MPI_Init(&argc, &argv);

	double start_time = MPI_Wtime();
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	
	printf("Hello world from process %d from total number of %d\n", rank, rank_size);

	double end_time = MPI_Wtime();

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (rank == 0) {
		printf("Execution time: %f seconds\n", end_time - start_time);
	}
	return 0;
}
