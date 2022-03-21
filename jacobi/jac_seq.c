#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAXSIZE 1000
#define MAXITERS 1000000

void jacobi(double** a, double** b);

int size, iters, workers;
double maxdiff = 0.0;


void print_to_file(double** grid){
    FILE* output_file = fopen("jacobi_seq.txt", "w");
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

    if(size > MAXSIZE)
        size = MAXSIZE;
    if(iters > MAXITERS) 
        iters = MAXITERS;
    
    size = size  + 2; //to include outer boundaries
    /*allocate memory  for both grid and new_grid */
    double** grid = malloc(size*sizeof(double*));
    double** new_grid = malloc(size*sizeof(double*));

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
    jacobi(grid, new_grid);
    clock_t end_time = clock();
    print_to_file(new_grid);
    printf("%g\n", maxdiff);
    double sec_elapsed = (double) (end_time-start_time)/CLOCKS_PER_SEC;
    printf("%g secs spent\n",sec_elapsed);
    free(grid);
    free(new_grid);

    return 0;
}

void jacobi(double** grid, double** new_grid){
    
    for(int iter_counter = 0; iter_counter < iters; iter_counter+=2){
        for(int i = 1; i  < size-1; i++){
            for( int j = 1; j < size-1; j++){
                new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]) * 0.25; //multiply by 0.25 instead dividing by 4
            }
        }

        for(int i = 1; i  < size-1; i++){
            for( int j = 1; j < size-1; j++){
                grid[i][j] = (new_grid[i-1][j] + new_grid[i+1][j] + new_grid[i][j-1] + new_grid[i][j+1]) * 0.25; //multiply by 0.25 instead dividing by 4
            }
        }

    }

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            maxdiff = fmax(maxdiff, fabs(grid[i][j] - new_grid[i][j]));
        }
    }
}
