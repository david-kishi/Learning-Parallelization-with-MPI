/******************************************************************************
* FILE: simple_cuda.cu
* DESCRIPTION:
*   A simple cuda program working with arrays.
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 25/03/2020 05:02:30 GMT-7
******************************************************************************/
#include <cuda.h>
#include <math.h>
#include <stdio.h>

#define     BLOCK_SIZE  1024



// Global function to set zeros
__global__
void zeroSet(int *d_array, int N){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N) {d_array[id] = 0;}
}


// Global function for computation
__global__
void addI(int *d_array, int N){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N) {d_array[id] += id;}
}


int main(int nn, char *str[]){
    // Variable declaration
    int *array, *d_array;
    unsigned N = atoi(str[1]); // gets user inputted N for array size

    // Determine number of blocks and blocksize
    // if N <= 1024, then nBlocks = 1 & nThreads = N
    unsigned nBlocks = ceil((float)N/BLOCK_SIZE); // number of blocks needed
    unsigned nThreads = N < BLOCK_SIZE ? N : BLOCK_SIZE; // determine threads

    // Memory Allocation
    array = (int*)malloc(N*sizeof(int));
    cudaMalloc(&d_array, N*sizeof(int));

    // Copy Contents of Host Array to Device Array
    cudaMemcpy(d_array, array, N*sizeof(int), cudaMemcpyHostToDevice);

    // Print array before zero-setting to check values
    printf("BEFORE ZERO-SETTING\n");
    for(int i = 0; i < N; i++){
        printf(" %d", array[i]);
    }
    printf("\n");

    // Call device function to set zeros then copy back to host memory
    zeroSet<<<nBlocks, nThreads>>>(d_array, N);
    cudaMemcpy(array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Print array after zero-setting to check values
    printf("AFTER ZERO_SETTING\n");
    for(int i = 0; i < N; i++){
        printf(" %d", array[i]);
    }
    printf("\n");

    // Call device function to compute then copy back to host memory
    addI<<<nBlocks, nThreads>>>(d_array, N);
    cudaMemcpy(array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Print array after computation to check values
    printf("AFTER ADDING I\n");
    for(int i = 0; i < N; i++){
        printf(" %d", array[i]);
    }
    printf("\n");

    // Free the memory
    cudaFree(d_array);
    free(array);

    return 0;
}