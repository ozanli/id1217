#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAXSIZE 1000
#define MAXITERS 1000000
#define MAXWORKERS 10

void jacobi(double** a, double** b, double*);

int size, iters, workers;
int height;

void print_to_file(double** grid){
    FILE* output_file = fopen("jacobi_parallel.txt", "w");
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            fprintf(output_file, "%g, ", grid[i][j]);
        }
        fprintf(output_file, "\n");
    }

    fclose(output_file);
}

int main(int argc, char const *argv[]){
    size = (argc > 1) ? atoi(argv[1]) : MAXSIZE;
    iters  = (argc > 2) ? atoi(argv[2]) : MAXITERS;
    workers  = (argc > 3) ? atoi(argv[3]) : MAXWORKERS;

    if(size > MAXSIZE)
        size = MAXSIZE;
    if(iters > MAXITERS) 
        iters = MAXITERS;

    
    height = size / workers;
    size = size  + 2; //to include outer boundaries
    /*allocate memory  for both grid and new_grid */
    double** grid = malloc(size*sizeof(double*));
    double** new_grid = malloc(size*sizeof(double*));
    double* maxdiff = malloc(workers * sizeof(double));

    for(int i = 0; i < size; i++){
        grid[i] = malloc(size*sizeof(double));
        new_grid[i] =  malloc(size*sizeof(double));
    }

    /* initialize the matrices with 1.0 in outer boundary and
        with 0.0 in inner field */

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            if(i == 0 || j == 0 || i == size-1 || j == size-1){
                grid[i][j] = 1;    
                new_grid[i][j] = 1;
            }
            else{
                grid[i][j] = 0;
                new_grid[i][j] = 0;
            } 
        }
    }
    
    clock_t start_time = clock();
    omp_set_num_threads(workers);
    jacobi(grid, new_grid, maxdiff);
    clock_t end_time = clock();
    print_to_file(new_grid);

    double abs_max_diff = 0.0;
    for(int i = 0; i < workers; i++)
        if(maxdiff[i] > abs_max_diff) abs_max_diff = maxdiff[i];

    printf("%g\n", abs_max_diff);
    double sec_elapsed = (double) (end_time-start_time)/CLOCKS_PER_SEC;
    printf("%g secs spent\n",sec_elapsed);
    free(grid);
    free(new_grid);

    return 0;
}

void jacobi(double** grid, double** new_grid, double* maxdiff){
    #pragma omp parallel 
    {
    int tid = omp_get_thread_num();
    int firstRow = tid*height + 1;
    printf("firstRow: %d\n", firstRow);
    int lastRow = firstRow + height - 1;
    if(lastRow > size - 1) lastRow = size - 1;
    printf("lastRow: %d\n", lastRow);
    double mydiff = 0.0;
        
    printf("my tid: %d\n", tid);
    #pragma omp barrier
    for(int iter_counter = 0; iter_counter < iters; iter_counter+=2){
        for(int i = firstRow; i <= lastRow; i++){
            for( int j = 1; j < size - 1 ; j++){
                printf("tid: %d, i: %d, j: %d\n", tid, i, j);
                new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]) * 0.25; //multiply by 0.25 instead dividing by 4

            }
        }
        
        #pragma omp barrier
        for(int i = firstRow; i  <= lastRow; i++){
            for( int j = 1; j < size-1; j++){
                grid[i][j] = (new_grid[i-1][j] + new_grid[i+1][j] + new_grid[i][j-1] + new_grid[i][j+1]) * 0.25; //multiply by 0.25 instead dividing by 4
            }
        }

        #pragma omp barrier
    }

    for(int i = firstRow; i <= lastRow; i++){
        for(int j = 0; j < size-1; j++){
            if((fabs(grid[i][j] - new_grid[i][j]) > mydiff))
                mydiff = (fabs(grid[i][j] - new_grid[i][j]));
        }
    }
    maxdiff[tid] = mydiff;
    #pragma omp barrier
    }
}
