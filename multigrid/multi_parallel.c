#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAXSIZE 10000
#define MAXITERS 1000000
#define MAXWORKERS 10

void jacobi(double** a, double** b, int size, int iters, int final);


int size, iters, workers;
double abs_max_diff = 0.0;

void restriction(double** fine, double** coarse, int coarse_size){
    int i;
    #pragma omp parallel for
    for(i = 1; i < coarse_size-1; i++){
        int x = i << 1;
        
        for(int j = 1; j < coarse_size-1; j++){
            int y = j << 1;
            coarse[i][j] = fine[x][y]*0.5 +(fine[x-1][y]+fine[x][y-1]+fine[x][y+1]+fine[x+1][y])*0.125;
        }
    }

}

void interpolation(double** fine, double** coarse, int fine_size, int coarse_size){
    /*udpate fine points*/
    int i, j, x, y;

    #pragma omp parallel
    {
        #pragma omp for private(x, y, j)
        for(i = 1; i < coarse_size-1; i++){
            x = i << 1;
            for(j = 1; j < coarse_size-1; j++){
                y = j << 1;
                fine[x][y] = coarse[i][j];
            }
        }

        #pragma omp for private(x, y, j)
        for(i = 1; i < fine_size-1; i+=2){
            for(j = 2; j < fine_size-1; j+=2){
                fine[i][j] = (fine[i-1][j]+fine[i+1][j]) * 0.5;
            }
        }

        #pragma omp for private(x, y, j)
        for(i = 1; i < fine_size-1; i++){
            for(j = 1; j < fine_size-1; j+=2){
                fine[i][j] = (fine[i][j-1]+fine[i][j+1]) * 0.5;
            }
        }
    }
}


void print_to_file(double** grid, int size){
    FILE* output_file = fopen("multi_parallel.txt", "w");
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
    workers = (argc > 3) ? atoi(argv[3]) : MAXWORKERS;

    if(size > MAXSIZE)
        size = MAXSIZE;
    if(iters > MAXITERS) 
        iters = MAXITERS;
    if(workers > MAXWORKERS)
        workers = MAXWORKERS;


    /*size => coarsest matrix (smallest)*/
    int size2 = (size * 2) + 1;
    int size3 = (size2 * 2) + 1;
    int size4 = (size3 * 2) + 1;
    
    size = size  + 2; //to include outer boundaries
    size2 = size2 + 2;
    size3 = size3 + 2;
    size4 = size4 + 2;

    /*allocate memory  for both grid and new_grid */
    double** grid = malloc(size*sizeof(double*));
    double** new_grid = malloc(size*sizeof(double*));

    double** grid2 = malloc(size2*sizeof(double*));
    double** new_grid2 = malloc(size2*sizeof(double*));

    double** grid3 = malloc(size3*sizeof(double*));
    double** new_grid3 = malloc(size3*sizeof(double*));

    double** grid4 = malloc(size4*sizeof(double*));
    double** new_grid4 = malloc(size4*sizeof(double*));

    double* maxdiff = malloc(workers * sizeof(double));

    for(int i = 0; i < size; i++){
        grid[i] = malloc(size*sizeof(double));
        new_grid[i] =  malloc(size*sizeof(double));
    }

    for(int i = 0; i < size2; i++){
        grid2[i] = malloc(size2*sizeof(double));
        new_grid2[i] =  malloc(size2*sizeof(double));
    }

    for(int i = 0; i < size3; i++){
        grid3[i] = malloc(size3*sizeof(double));
        new_grid3[i] =  malloc(size3*sizeof(double));
    }

    for(int i = 0; i < size4; i++){
        grid4[i] = malloc(size4*sizeof(double));
        new_grid4[i] =  malloc(size4*sizeof(double));
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
 
    for(int i = 0; i < size2; i++){
        for(int j = 0; j < size2; j++){
            if(i == 0 || j == 0 || i == size2-1 || j == size2-1){
                grid2[i][j] = 1;    
                new_grid2[i][j] = 1;
            }
            else{
                grid2[i][j] = 0;
                new_grid2[i][j] = 0;
            } 
        }
    }

    for(int i = 0; i < size3; i++){
        for(int j = 0; j < size3; j++){
            if(i == 0 || j == 0 || i == size3-1 || j == size3-1){
                grid3[i][j] = 1;    
                new_grid3[i][j] = 1;
            }
            else{
                grid3[i][j] = 0;
                new_grid3[i][j] = 0;
            } 
        }
    }

    for(int i = 0; i < size4; i++){
        for(int j = 0; j < size4; j++){
            if(i == 0 || j == 0 || i == size4-1 || j == size4-1){
                grid4[i][j] = 1;    
                new_grid4[i][j] = 1;
            }
            else{
                grid4[i][j] = 0;
                new_grid4[i][j] = 0;
            } 
        }
    }

    omp_set_num_threads(workers);
    double start_time = omp_get_wtime();

    /*go down in V*/
    jacobi(grid4, new_grid4, size4,  4, 0);
    restriction(grid4, grid3, size3);


    jacobi(grid3, new_grid3, size3, 4, 0);
    restriction(grid3, grid2, size2);

    jacobi(grid2, new_grid2, size2, 4, 0);
    restriction(grid2, grid, size);

    /*bottom reached, go up from V*/


    jacobi(grid, new_grid, size, iters, 0);
    interpolation(grid2, grid, size2, size);

    jacobi(grid2, new_grid2, size2, 4, 0);
    interpolation(grid3, grid2, size3, size2);

    jacobi(grid3, new_grid3, size3, 4, 0);
    interpolation(grid4, grid3, size4, size3);

    jacobi(grid4, new_grid4, size4, 4, 1);

    double end_time = omp_get_wtime();


    print_to_file(grid4, size4);
    printf("size: %d\n", size-2);
    printf("iters: %d\n", iters);
    printf("maxDiff: %g\n", abs_max_diff);
    printf("workers: %d\n", workers);
    double sec_elapsed = end_time-start_time;
    printf("%g secs spent\n",sec_elapsed);
    free(grid);
    free(new_grid);

    free(grid2);
    free(new_grid2);

    free(grid3);
    free(new_grid3);


    free(grid4);
    free(new_grid4);

    return 0;
}

void jacobi(double** grid, double** new_grid, int size, int iters, int final){
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double mydiff = 0.0;

        #pragma omp barrier
        for(int iter_counter = 0; iter_counter < iters; iter_counter+=2){
            int i, j;
            #pragma omp for 
            for(i = 1; i  < size-1; i++){
                for(j = 1; j < size-1; j++){
                    new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]) * 0.25; //multiply by 0.25 instead dividing by 4
                }
            }

            #pragma omp for
            for(i = 1; i  < size-1; i++){
                for(j = 1; j < size-1; j++){
                    grid[i][j] = (new_grid[i-1][j] + new_grid[i+1][j] + new_grid[i][j-1] + new_grid[i][j+1]) * 0.25; //multiply by 0.25 instead dividing by 4
                }
            }

        }

        if(final){
            int i;
            #pragma omp for 
            for(i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                    if((fabs(grid[i][j] - new_grid[i][j]) > abs_max_diff))
                        #pragma omp critical
                        abs_max_diff = (fabs(grid[i][j] - new_grid[i][j]));
                }
            }
        }
    }
}
