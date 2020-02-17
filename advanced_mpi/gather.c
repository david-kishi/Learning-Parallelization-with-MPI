#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) 
{
    int n = 5;  // 5
    int rank, size;
    int a[5];  
    int i;    

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int b = rank;

    for (i = 0; i < n ; i++) //n = 5
        a[i] = 0;
    printf("Process of rank %d has array [%d, %d, %d, %d %d] and b = %d \n", rank, a[0], a[1], a[2], a[3], a[4], b);

    MPI_Gather(&b, 1, MPI_INT, &a, 1, MPI_INT, 2, MPI_COMM_WORLD);

    printf("Process of rank %d has array [%d, %d, %d, %d %d] and b = %d \n", rank, a[0], a[1], a[2], a[3], a[4], b);

    MPI_Finalize();

    return 0;
}