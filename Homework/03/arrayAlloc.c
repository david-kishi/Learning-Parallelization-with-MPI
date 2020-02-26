/******************************************************************************
* FILE: arrayAlloc.c
* DESCRIPTION:
*   Display the indexes of the array allocated to each process.
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 26/02/2020
******************************************************************************/
#include    <stdio.h>
#include    <stdlib.h>
#include    <time.h>
#include    <mpi.h>

// Function to randomly generate array of odd numbers
int randomInt(){
    int temp = rand() % 10;
    if ( temp == 0 ){ temp += 1; }
    else if ( temp % 2 == 0 ){ temp -= 1; }

    return temp;
}

int main(int argc, char *argv[])
{
    // Seed random number generator
    time_t t;
    srand((unsigned) time(&t));

    int     rank,   // Process rank number
            size,   // Number of processes
            sbuf,   // Send Buffer
            rbuf = 0;   // Receive Buffer
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    sbuf = randomInt();

    MPI_Scan(&sbuf, &rbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("process %d will receive the array portion between index %d-%d\n", rank, rbuf-sbuf+1, rbuf);
    
    MPI_Finalize();
}