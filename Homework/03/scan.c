/******************************************************************************
* FILE: scan.c
* DESCRIPTION:
*   inclusive scan operation utilizing mpi scatter
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 25/02/2020
******************************************************************************/
#include    <stdio.h>
#include    <mpi.h>

#define     MINIMUM_PROC    8
#define     SCATTER_SIZE    1

//TODO - Create function to randomly generate array of odd numbers


int main(int argc, char *argv[])
{
    int     arr[] = {3, 1, 7, 1, 5, 1, 5, 3};
    int     rank,       // Process rank number
            size,       // Number of processes
            buf = 0;    // buffer
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if processor count requirement is fulfilled
    if(size != MINIMUM_PROC)
    {
        if(rank==0)
        {
            printf("Quitting. Minimum processes is %d. Current: %d\n", MINIMUM_PROC, size);
        }
    }
    else
    {
        printf("Process of rank %d has array of [%d, %d, %d, %d, %d, %d, %d, %d] \n", rank, arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);

        
        MPI_Scatter(arr, 1, MPI_INT, &buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

        

        printf("Process of rank %d has b value of %d\n", rank, buf);

        // MPI_Gather()
    }
    


    MPI_Finalize();
}