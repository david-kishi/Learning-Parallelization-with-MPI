# CUDA MATRIX SQUARING

## Instructions
Build program with CUDA.
`nvcc cuda_matrix.cu`

Parameters `-o <name of output file>` are optional. No parameters will default to `a.out`.

### How to run
`./<name of executable>`

Example:
`./a.out`

### How program works
1. Execute.
2. Two matrices on host are declared. One for computing and one empty to later
hold result matrix.
3. Matrices on host are allocated memory based on N*N size.
4. Computing matrix is populated with a global value.
5. Output computing matrix.
6. Call host function to initiate Matrix Squaring.
7. Matrices declared to hold host matrices.
6. Copy contents of host computing matrix to device computing matrix.
8. Call device function for Matrix Squaring.
9. Copy contents of device result matrix to host result matrix.
10. Free device memory.
10. Output squared matrix.
11. Free host memory
12. Fin.

# FORTRAN MATRIX ADDITION

## Instructions
Build program with gfortran compiler.
`gfortran fortran_matrix.f90`

Parameters `-o <name of output file>` are optional. No parameters will default
to `a.out`.

### How to run
`./<name of executable>`

Example:
`./a.out`

### How program works
1. Matrices A, B, and C are declared with dimension size 4x4.
2. Set all values of A to 4.
3. Loop through B, setting each value to i+j+1.
4. Loop through C, setting each value to A(j,i) + B(j,i) while checking if
result is equal to 8. If so, replace with 16.
5. Print matrices A, B, and C by calling subroutine printMatrix.