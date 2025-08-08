#include <stdio.h>
#include <omp.h>

double compute_pi(int num_steps, int num_threads) 
{
    double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (int i = 0; i < num_steps; i++) 
    {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}

int main() {
    int num_steps;
    int num_threads;

    printf("Enter number of steps: ");
    scanf("%d", &num_steps);
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    double start = omp_get_wtime();
    double pi = compute_pi(num_steps, num_threads);
    double end = omp_get_wtime();

    printf("Calculated Pi = %.5f\n", pi);
    printf("Execution Time: %f seconds\n", end - start);

    return 0;
}