/******************************************************************************
* FILE: array_loops.c
* DESCRIPTION:
*   Initialize array of size 64 in parallel, then add 3*i to each element,
*   then counts amount of odd values in array
* AUTHOR: David Nguyen
* CONTACT: david@knytes.com
* LAST REVISED: 04/03/2020
******************************************************************************/
#include <omp.h>
#include <stdio.h>

#define     DESIRED_ARR_SIZE    64
#define     DESIRED_NUM_THREADS 8

int main (int nargs , char ** args) {
    int     array[DESIRED_ARR_SIZE], // array to hold values
            i, // for loop iterator
            sum = 0, // master odd count sum
            oddCount; // local odd count sum

    // Sets omp threads to DESIRED_NUM_THREADS if not already
    omp_set_num_threads(DESIRED_NUM_THREADS);

    // Populate array with all zeros
    #pragma omp parallel for
        for (i = 0; i < DESIRED_ARR_SIZE; i++){
            array[i] = 0;
        }

    // Add 3*i to each element of array
    #pragma omp parallel for
        for (i = 0; i < DESIRED_ARR_SIZE; i++){
            array[i] += 3*i;           
        }

    #pragma omp parallel private(oddCount) shared(sum)
    {
        int t_id = omp_get_thread_num();

        // set oddCount to 0. Fixes memory corruption issue.
        oddCount = 0;

        for (i = t_id; i < DESIRED_ARR_SIZE; i += DESIRED_NUM_THREADS){
            if(array[i]%2 != 0){
                oddCount += 1;
            }
        }
        
        #pragma omp critical
        sum += oddCount;
    }

    // Print number of odd numbers
    printf("Count of odd numbers: %d\n", sum);

    return 0;
}