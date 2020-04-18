# Instructions
Build program with CUDA.
`nvcc simple_cuda.cu`

Parameters `-o <name of output file>` are optional. No parameters will default to `a.out`.

## How to run
Run with specified array size.

`./<name of executable> <array size>`

Example:
`./a.out 16`

If no array size is specified, segmentation fault will occur.

## How program works
1. Execute.
2. Host and device array is created set.
3. Host and device array are allocated memory of size based on user specified.
4. Copy contents of host array to device array.
5. Call device function to populate array with zeroes.
6. Copy contents of device array to host array.
7. Output
8. Call device function to populate array with index number.
9. Copy contents of device array to host array.
10. Output
11. Free memory & fin.