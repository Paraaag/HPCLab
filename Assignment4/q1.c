#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n;
    printf("Enter the number of terms in the Fibonacci series: ");
    scanf("%d", &n);

    long long* v = (long long*)malloc((n + 1) * sizeof(long long));
    if (v == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    v[0] = 0;
    v[1] = 1;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Computing series using %d threads\n", omp_get_num_threads());
        }

        #pragma omp for schedule(static) nowait
        for (int i = 2; i <= n; i++) {
            #pragma omp critical
            {
                v[i] = v[i - 1] + v[i - 2];
            }
        }
    }

    double end = omp_get_wtime();

    printf("series: ");
    for (int i = 0; i <= n; i++)
        printf("%lld ", v[i]);
    printf("\nTime taken: %f sec\n", end - start);

    free(v);
    return 0;
}
