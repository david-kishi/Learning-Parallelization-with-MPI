# Instructions
Run program as a normal MPI program as taught in class.

## How to run
Run with specified amount of processes.

`mpirun -n <# of processes> ./<name of executable>`

Run with max amount of processes available on system.

`mpirun ./<name of executable>`

## How program works
1. Execute.
2. Array is created set at the desired size required.
3. OMP threads are set in case not set already in terminal.
4. Parallel for loop initiated to set all elements in array to 0.
5. Parallel for loop initiated to add 3*index to all elements in array.
6. Parallel construct initiated with a cyclic distributed for loop to traverse
    through the array looking for odd numbers.
7. Print out amount of odd numbers found.