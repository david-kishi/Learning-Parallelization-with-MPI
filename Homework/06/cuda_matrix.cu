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
#define NUM         10



// Global function for Matrix Squaring
__global__
void deviceMatrixSquare(unsigned *mat_d, unsigned *p_mat_d, int width){
    unsigned p_value = 0;
    for(int k = 0; k < width; ++k){
        unsigned element = mat_d[threadIdx.y*width+k];
        p_value = element * element;
        p_mat_d[threadIdx.y*width+threadIdx.x] = p_value;
    }
}

// Host function to process 
__host__
void matrixSquare(unsigned *mat, unsigned *p_mat){
    unsigned size = N*N*sizeof(unsigned);
    unsigned *mat_d, *p_mat_d;

    // Determine number of blocks and blocksize
    // if N <= 1024, then nBlocks = 1 & nThreads = N
    unsigned nBlocks = ceil(((float)N*N)/BLOCK_SIZE); // number of blocks needed
    unsigned nThreads = N*N <= BLOCK_SIZE ? N*N : BLOCK_SIZE; // determine threads

    // Memory Allocation
    cudaMalloc(&mat_d, size);
    cudaMalloc(&p_mat_d, size);

    // Load mat_d to device memory
    cudaMemcpy(mat_d, mat, size, cudaMemcpyHostToDevice);

    // Device Function - Matrix Squaring
    deviceMatrixSquare<<<nBlocks, nThreads>>>(mat_d, p_mat_d, N);

    // Copy result from device memory
    cudaMemcpy(p_mat, p_mat_d, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(mat_d);
    cudaFree(p_mat_d);
}

int main(){
    // Variable declaration
    unsigned *mat, *p_mat;

    // Memory Allocation
    mat = (unsigned*)malloc(N*N*sizeof(unsigned));
    p_mat = (unsigned*)malloc(N*N*sizeof(unsigned));

    // Set all matrix values to NUM
    for(int i = 0; i < N*N; i++){
        mat[i] = NUM;
    }

    // Print array after num-setting to check values
    printf("Matrix\n");
    for(int i = 0; i < N*N; i++){
        printf(" %d", mat[i]);
        // Newline if end of row
        if((i+1)%N==0){
            printf("\n");
        }
    }
    printf("\n");

    // Call matrix squaring function
    matrixSquare(mat, p_mat);

    // Print array after num-setting to check values
    printf("AFTER MATRIX SQUARING\n");
    for(int i = 0; i < N*N; i++){
        printf(" %d", p_mat[i]);
        // Newline if end of row
        if((i+1)%N==0){
            printf("\n");
        }
    }
    printf("\n");

    // Free memory
    free(mat);

    return 0;
}