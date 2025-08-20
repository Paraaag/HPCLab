#include <stdio.h>
#include <omp.h>

long long find(int n) {
    if (n < 2) return n;

    long long x, y;

    #pragma omp task shared(x)
    x = find(n - 1);

    #pragma omp task shared(y)
    y = find(n - 2);

    #pragma omp taskwait
    return x + y;
}

int main() {
    int n;
    printf("Enter n: ");
    scanf("%d", &n);

    long long result;
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = find(n);
        }
    }

    double end = omp_get_wtime();

    printf("Fibonacci(%d) = %lld\n", n, result);
    printf("Time taken: %f sec\n", end - start);

    return 0;
}
