#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_add(int size) 
{
    int i, j;
    double **a = malloc(size * sizeof(double *));
    double **b = malloc(size * sizeof(double *));
    double **c = malloc(size * sizeof(double *));

    for (i = 0; i < size; i++) 
    {
        a[i] = malloc(size * sizeof(double));
        b[i] = malloc(size * sizeof(double));
        c[i] = malloc(size * sizeof(double));
        for (j = 0; j < size; j++) 
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (i = 0; i < size; i++) 
    {
        for (j = 0; j < size; j++) 
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }

    double end = omp_get_wtime();
    printf("Size: %d, Time: %f seconds\n", size, end - start);

    printf("Size: %d, Time: %f seconds\n", size, end - start);
    for (i = 0; i < size; i++) 
    {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);
}

int main() {
    int sizes[] = {250, 500, 750, 1000, 2000};
    for (int i = 0; i < 5; i++) {
        matrix_add(sizes[i]);
    }
    return 0;
}