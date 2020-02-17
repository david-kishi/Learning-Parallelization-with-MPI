#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    int rank;    
    int size;        
    
    MPI_Init(&argc,&argv);     
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);    
    MPI_Comm_size(MPI_COMM_WORLD,&size);  

    if (rank == 0) {
    rand_nums = create_rand_nums(elements_per_proc * world_size);
    }

    // Create a buffer that will hold a subset of the random numbers
    float *sub_rand_nums = malloc(sizeof(float) * elements_per_proc);

    // Scatter the random numbers to all processes
    MPI_Scatter(rand_nums, elements_per_proc, MPI_FLOAT, sub_rand_nums, elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the average of your subset
    float sub_avg = compute_avg(sub_rand_nums, elements_per_proc);
    
    // Gather all partial averages down to the root process
    float *sub_avgs = NULL;
    if (rank == 0) {
        sub_avgs = malloc(sizeof(float) * world_size);
    }
    MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the total average of all numbers.
    if (rank == 0) {
        float avg = compute_avg(sub_avgs, world_size);
    }

    return 0;
}