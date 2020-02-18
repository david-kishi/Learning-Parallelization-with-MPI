/******************************************************************************
* FILE: xchg_dbl.c
* DESCRIPTION:
*   Exchange one double value between process with rank 0 and process with
*   rank 1. Calculate the execution time using MPI_Wtime
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 18/02/2020
******************************************************************************/
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
}