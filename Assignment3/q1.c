#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int i;
    int SIZE;
    printf("Enter Size: ");
    scanf("%d",&SIZE);
    int *a = malloc(SIZE * sizeof(int));
    int *b = malloc(SIZE * sizeof(int));
    long long dot_product = 0;
    
    for (i = 0; i < SIZE; i++) {
        a[i] = i % 100;
        b[i] = (SIZE - i) % 100;
    }

    double start = omp_get_wtime();
    #pragma omp parallel for reduction(+:dot_product)
    for (i = 0; i < SIZE; i++) {
        dot_product += a[i] * b[i];
    }

    double end = omp_get_wtime();
    printf("Minimum Scalar Product: %lld\n", dot_product);
    printf("Time Taken: %f seconds\n", end - start);
    free(a);
    free(b);
    return 0;
}