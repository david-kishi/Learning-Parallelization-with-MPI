#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    char name[20];
    printf("What is your name \n");
    scanf("%s", name);
    printf("Your name is %s \n", name);

    MPI_Finalize();
}