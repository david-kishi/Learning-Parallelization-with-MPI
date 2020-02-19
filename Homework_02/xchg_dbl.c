/******************************************************************************
* FILE: xchg_dbl.c
* DESCRIPTION:
*   Exchange one double value between process with rank 0 and process with
*   rank 1. Calculate the execution time using MPI_Wtime.
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 18/02/2020
******************************************************************************/
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    double  dblVal,     // Value to be exchanged
            blkTimeStart,       // Holds MPI_Wtime() start for blocking prcd.
            blkTimeStop,        // Holds MPI_Wtime() stop for blocking prcd.
            nonBlkTimeStart,    // Holds MPI_Wtime() start for non-blocking.
            nonBlkTimeStop;     // Holds MPI_Wtime() stop for non-blocking.
    int     rank,       // Process rank number
            size;       // Number of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request ireqs;
    MPI_Status istatus;

    /* BLOCKING COMMANDS */
    if (rank == 0)
    {
        dblVal = 5.11;  // Assign dblVal a float value
        blkTimeStart = MPI_Wtime();
        MPI_Send(&dblVal, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent double value %f to process 1\n", dblVal);
    }
    else if (rank == 1)
    {
        MPI_Recv(&dblVal, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        blkTimeStop = MPI_Wtime();
        printf("Process 1 received double value %f from process 0\n", dblVal);
        printf("BLOCKING TIME:\t %f\n", (blkTimeStop - blkTimeStart));
        printf("%f\n", blkTimeStop);
        printf("%f\n", blkTimeStart);
    }

    /* NON-BLOCKING COMMANDS */    
    if (rank == 0)
    {
        nonBlkTimeStart = MPI_Wtime();
        dblVal = 7.14;  // Assign dblVal a different float value from blocking
        MPI_Isend(&dblVal, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &ireqs);
        MPI_Wait(&ireqs, &istatus);
        printf("Process 0 sent double value %f to process 1\n", dblVal);
    }
    else if (rank == 1)
    {
        MPI_Irecv(&dblVal, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &ireqs);
        MPI_Wait(&ireqs, &istatus);
        nonBlkTimeStop = MPI_Wtime();
        printf("Process 1 received double value %f from process 0\n", dblVal);
        printf("NON-BLOCKING TIME:\t %f\n", (nonBlkTimeStop - nonBlkTimeStart));
        printf("%f\n", nonBlkTimeStop);
        printf("%f\n", nonBlkTimeStart);
    }

    MPI_Finalize();
}