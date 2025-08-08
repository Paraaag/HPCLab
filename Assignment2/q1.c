// vector_scalar_add.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void vector_scalar_addition(float *a, float scalar, int n, int num_threads) 
{
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) 
    {
        a[i] += scalar;
    }
}

int main() {
    int n;
    float scalar;
    int num_threads;

    printf("Enter size of vector: ");
    scanf("%d", &n);
    printf("Enter scalar value to add: ");
    scanf("%f", &scalar);
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    float *a = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) 
    {
        a[i] = i;
    }

    double start = omp_get_wtime();
    vector_scalar_addition(a, scalar, n, num_threads);
    double end = omp_get_wtime();

    printf("Execution Time: %f seconds\n", end - start);

    free(a);
    return 0;
}