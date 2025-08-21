#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 2048

static int A[MAX_SIZE][MAX_SIZE];
static int B[MAX_SIZE][MAX_SIZE];

double matrix_scalar_mul(int size, int scalar, int sched_kind, int chunk) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            A[i][j] = i + j;

    int sum = 0; 

    double start_time = omp_get_wtime();

    if (sched_kind == 1) { 
        #pragma omp parallel for collapse(2) schedule(static,1) reduction(+:sum)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                B[i][j] = A[i][j] * scalar;
                sum += B[i][j];
            }
        }
    } 
    else if (sched_kind == 2) { 
        #pragma omp parallel for collapse(2) schedule(dynamic,chunk) reduction(+:sum)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                B[i][j] = A[i][j] * scalar;
                sum += B[i][j];
            }
        }
    } 
    else { 
        #pragma omp parallel for collapse(2) schedule(guided,chunk) reduction(+:sum)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                B[i][j] = A[i][j] * scalar;
                sum += B[i][j];
            }
        }
    }

    return omp_get_wtime() - start_time;
}

int main() {
    int sizes[] = {512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int max_threads, scalar = 5, sched_kind, chunk;
    double time_sequential, time_parallel, speedup;

    printf("Enter the number of threads : ");
    scanf("%d", &max_threads);
    printf("Choose schedule (1=static, 2=dynamic, 3=guided): ");
    scanf("%d", &sched_kind);
    printf("Enter schedule chunk size (recommend 8 or 16, 0 for default): ");
    scanf("%d", &chunk);

    printf("\n*** Time vs. Data Size (Threads fixed at %d) ***\n", max_threads);
    omp_set_num_threads(max_threads);
    for (int i = 0; i < num_sizes; i++) {
        time_parallel = matrix_scalar_mul(sizes[i], scalar, sched_kind, chunk);
        printf("Size: %d x %d, Threads: %d, Time: %f s\n", sizes[i], sizes[i], max_threads, time_parallel);
    }

    int fixed_size = 2048;
    printf("\n*** Time vs. Threads ***(Size fixed at %d x %d) ---\n", fixed_size, fixed_size);

    omp_set_num_threads(1);
    time_sequential = matrix_scalar_mul(fixed_size, scalar, sched_kind, chunk);
    printf("Threads: 1, Time: %f s (Baseline)\n", time_sequential);

    for (int t = 2; t <= max_threads; t *= 2) {
        omp_set_num_threads(t);
        time_parallel = matrix_scalar_mul(fixed_size, scalar, sched_kind, chunk);
        speedup = time_sequential / time_parallel;
        printf("Threads: %d, Time: %f s, Speedup: %.2fx\n", t, time_parallel, speedup);
    }

    return 0;
}