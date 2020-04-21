all: complex_mul

# Define Fortran compiler
FC= ifort
#FC= g95

complex_mul: main.f90 Cuda_function.o
	$(FC) -o complex_mul main.f90 Cuda_function.o -L/usr/local/cuda/lib  -lcudart

Cuda_function.o: Cuda_function.cu
	nvcc -c -O3 Cuda_function.cu

clean: 
	rm complex_mul Cuda_function.o
