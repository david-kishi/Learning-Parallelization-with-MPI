/******************************************************************************
* FILE: xchg_dbl.c
* DESCRIPTION:
*   Exchange one double value between process with rank 0 and process with
*   rank 1. Calculate the execution time using MPI_Wtime.
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 19/02/2020
******************************************************************************/
#include    <stdio.h>
#include    <mpi.h>

#define     MINIMUM_PROC    4

int main(int argc, char *argv[])
{
    /*  dblVal____[4]
        0: Exchange Value
        1: Start Time
        2: Stop Time Single Transmission
        3: Stop Time Round Trip Transmission
    */
    double  dblValBlk[4],
            dblValNonBlk[4];
    int     rank,           // Process rank number
            size;           // Number of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request ireqs;
    MPI_Status istatus;

    /* Check if there's enough processes to run entire program */
    if(size < 4)
    {
        if(rank==0)
        {
            printf("Quitting. Minimum processes is %d. Current: %d\n", MINIMUM_PROC, size);
        }

    }
    else
    {
        /* BLOCKING COMMANDS */
        if (rank == 0)
        {
            dblValBlk[0] = 5.11; // Assign dblVal a float value
            dblValBlk[1] = MPI_Wtime(); // Holds start time to pass through MPI_SEND for cleaner output

            MPI_Send(&dblValBlk, 4, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            printf("Process 0 sent double value %f to process 1\n", dblValBlk[0]);

            MPI_Recv(&dblValBlk, 4, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dblValBlk[3] = MPI_Wtime();
            printf("Process 0 received double value %f from process 1\n", dblValBlk[0]);

            /* Print Blocking Send/Receive Execution Time - Round-trip */
            printf("\tBLOCKING TIME (Round-trip): %f\n", dblValBlk[3] - dblValBlk[1]);
        }
        else if (rank == 1)
        {
            MPI_Recv(&dblValBlk, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dblValBlk[2] = MPI_Wtime();
            printf("Process 1 received double value %f from process 0\n", dblValBlk[0]);
            
            dblValBlk[0] += 1; // Change exchange value to clearly see round-trip transmission
            MPI_Send(&dblValBlk, 4, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            printf("Process 1 sent double value %f to process 0\n", dblValBlk[0]);

            /* Print Blocking Send/Receive Execution Time - Single Transmission */
            printf("\tBLOCKING TIME (Single-transmission): %f\n", dblValBlk[2] - dblValBlk[1]);
        }
        
        /* NON-BLOCKING COMMANDS */    
        else if (rank == 2)
        {
            dblValNonBlk[0] = 7.14; // Assign dblVal a different float value from blocking
            dblValNonBlk[1] = MPI_Wtime(); // Holds start time to pass through MPI_SEND for cleaner output

            MPI_Isend(&dblValNonBlk, 4, MPI_DOUBLE, 3, 2, MPI_COMM_WORLD, &ireqs);
            MPI_Wait(&ireqs, &istatus);
            printf("Process 2 sent double value %f to process 3\n", dblValNonBlk[0]);

            MPI_Irecv(&dblValNonBlk, 4, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD, &ireqs);
            MPI_Wait(&ireqs, &istatus);
            dblValNonBlk[3] = MPI_Wtime();
            printf("Process 2 received double value %f from process 3\n", dblValNonBlk[0]);

            /* Print Blocking Send/Receive Execution Time - Round-trip */
            printf("\tNON-BLOCKING TIME (Round-trip): %f\n", dblValNonBlk[3] - dblValNonBlk[1]);
        }
        else if (rank == 3)
        {
            MPI_Irecv(&dblValNonBlk, 4, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &ireqs);
            MPI_Wait(&ireqs, &istatus);
            dblValNonBlk[2] = MPI_Wtime();
            printf("Process 3 received double value %f from process 2\n", dblValNonBlk[0]);

            dblValNonBlk[0] += 1; // Change exchange value to clearly see round-trip transmission
            MPI_Isend(&dblValNonBlk, 4, MPI_DOUBLE, 2, 3, MPI_COMM_WORLD, &ireqs);
            MPI_Wait(&ireqs, &istatus);
            printf("Process 3 sent double value %f to process 2\n", dblValNonBlk[0]);

            /* Print Blocking Send/Receive Execution Time - Single Transmission */
            printf("\tNON-BLOCKING TIME (Single-transmission): %f\n", (dblValNonBlk[2] - dblValNonBlk[1]));
        }
    }

    MPI_Finalize();
}