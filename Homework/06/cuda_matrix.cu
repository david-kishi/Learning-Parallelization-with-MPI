/******************************************************************************
* FILE: cuda_matrix.cu
* DESCRIPTION:
*   A simple cuda program to compute square of a N dimensional matrix.
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 20/04/2020
******************************************************************************/
#include <cuda.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE  1024
#define N           32
#define NUM         4



// Host function to set values of matrix to one integer
__host__
void numSet(int *mat){
    for(int i = 0; i < N*N; i++){
        mat[i] = NUM;
    }
}


// Global function for computation
__global__
void addI(int *d_array, int N){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N) {d_array[id] += id;}
}


int main(){
    // Variable declaration
    unsigned *mat, *d_mat;

    // Determine number of blocks and blocksize
    // if N <= 1024, then nBlocks = 1 & nThreads = N
    // unsigned nBlocks = ceil(((float)N*N)/BLOCK_SIZE); // number of blocks needed
    // unsigned nThreads = N*N <= BLOCK_SIZE ? N*N : BLOCK_SIZE; // determine threads

    // Memory Allocation
    mat = (unsigned*)malloc(N*N*sizeof(unsigned));
    // cudaMalloc(&d_mat, N*N*sizeof(unsigned));

    // // Copy Contents of Host Array to Device Array
    // cudaMemcpy(d_mat, mat, N*N*sizeof(unsigned), cudaMemcpyHostToDevice);

    

    // Print array before zero-setting to check values
    printf("BEFORE NUM-SETTING\n");
    for(int i = 0; i < N*N; i++){
        printf(" %d", mat[i]);
    }
    printf("\n");

    numSet(mat);

    // // Call device function to set zeros then copy back to host memory
    // zeroSet<<<nBlocks, nThreads>>>(d_array, N);
    // cudaMemcpy(array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Print array after zero-setting to check values
    printf("AFTER NUM_SETTING\n");
    for(int i = 0; i < N*N; i++){
        printf(" %d", mat[i]);
    }
    printf("\n");

    // // Call device function to compute then copy back to host memory
    // addI<<<nBlocks, nThreads>>>(d_array, N);
    // cudaMemcpy(array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);

    // // Print array after computation to check values
    // printf("AFTER ADDING I\n");
    // for(int i = 0; i < N; i++){
    //     printf(" %d", array[i]);
    // }
    // printf("\n");

    // Free the memory
    cudaFree(d_array);
    free(array);

    return 0;
}