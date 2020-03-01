# Instructions
Run program as a normal MPI program as taught in class.

## How to run
Run with specified amount of processes.

`mpirun -n <# of processes> ./<name of executable>`

Run with max amount of processes available on system.

`mpirun ./<name of executable>`

## How program works
1. Execute
2. Declare buffers and initiate MPI
3. Random variable is assigned to send buffer `sbuf` utilizing function `randomInt()`
4. Execute MPI_Scan(...)
5. Print index
6. Steps 3-5 repeats till no more processes