#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 2048


static double A[MAX_SIZE][MAX_SIZE];
static double x[MAX_SIZE];
static double y[MAX_SIZE];

double matrix_vector_mul(int size, int sched_kind, int chunk) {
    for (int i = 0; i < size; i++) {
        x[i] = i * 0.5;
        for (int j = 0; j < size; j++) {
            A[i][j] = i + j;
        }
    }

    double start_time = omp_get_wtime();

    if (sched_kind == 1) { 
        #pragma omp parallel for schedule(static,1)
        for (int i = 0; i < size; i++) {
            y[i] = 0.0;
            for (int j = 0; j < size; j++) {
                y[i] += A[i][j] * x[j];
            }
        }
    } 
    else if (sched_kind == 2) { 
        #pragma omp parallel for schedule(dynamic,chunk)
        for (int i = 0; i < size; i++) {
            y[i] = 0.0;
            for (int j = 0; j < size; j++) {
                y[i] += A[i][j] * x[j];
            }
        }
    }
    else { 
        #pragma omp parallel for schedule(guided,chunk)
        for (int i = 0; i < size; i++) {
            y[i] = 0.0;
            for (int j = 0; j < size; j++) {
                y[i] += A[i][j] * x[j];
            }
        }
    }

    return omp_get_wtime() - start_time;
}

int main() {
    int sizes[] = {512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int max_threads, sched_kind, chunk;
    double time_sequential, time_parallel, speedup;

    printf("Enter the number of threads: ");
    scanf("%d", &max_threads);
    printf("Choose schedule (1=static, 2=dynamic, 3=guided): ");
    scanf("%d", &sched_kind);
    printf("Enter schedule chunk size (recommend 8 or 16, 0 for default): ");
    scanf("%d", &chunk);

    printf("\n*** Time vs. Data Size (Threads fixed at %d) ***\n", max_threads);
    omp_set_num_threads(max_threads);
    for (int i = 0; i < num_sizes; i++) {
        time_parallel = matrix_vector_mul(sizes[i], sched_kind, chunk);
        printf("Size: %d x %d, Threads: %d, Time: %f s\n", sizes[i], sizes[i], max_threads, time_parallel);
    }
    int fixed_size = 2048;
    printf("\n*** Time vs. Threads (Size fixed at %d x %d) ***\n", fixed_size, fixed_size);

    omp_set_num_threads(1);
    time_sequential = matrix_vector_mul(fixed_size, sched_kind, chunk);
    printf("Threads: 1, Time: %f s (Baseline)\n", time_sequential);

    for (int t = 2; t <= max_threads; t *= 2) {
        omp_set_num_threads(t);
        time_parallel = matrix_vector_mul(fixed_size, sched_kind, chunk);
        speedup = time_sequential / time_parallel;
        printf("Threads: %d, Time: %f s, Speedup: %.2fx\n", t, time_parallel, speedup);
    }

    return 0;
}