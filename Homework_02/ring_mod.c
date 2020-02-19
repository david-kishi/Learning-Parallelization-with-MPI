/******************************************************************************
* FILE: ring_mod.c
* DESCRIPTION:
*   Modified ring example to calculate execution time.
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 19/02/2020
******************************************************************************/
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    double  start,  // Start Time
            stop;   // Stop Time
    int     rank,   // Process rank number
            size;   // Number of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("I am %d of %d\n", rank, size);

    int token;
    if (rank != 0)
    {
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d from process %d\n", rank, token, rank - 1);
    }
    else
    {
        // Set the token's value if you are process 0
        token = 5;
    }
    MPI_Send(&token, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    // Now process 0 can receive from the last process.
    if (rank == 0)
    {
        start = MPI_Wtime(); // Start Timer

        MPI_Recv(&token, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        stop = MPI_Wtime(); // End Timer
        
        printf("Process %d received token %d from process %d\n", rank, token, size - 1);

        /* Print execution time */
        printf("\tExecution Time: %f\n", stop - start);
    }
    MPI_Finalize();
}