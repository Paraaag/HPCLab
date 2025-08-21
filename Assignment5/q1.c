#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void init_matrices(int n, int ***a, int ***b, int ***c) {
    *a = (int**) malloc(n * sizeof(int*));
    *b = (int**) malloc(n * sizeof(int*));
    *c = (int**) malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        (*a)[i] = (int*) malloc(n * sizeof(int));
        (*b)[i] = (int*) malloc(n * sizeof(int));
        (*c)[i] = (int*) malloc(n * sizeof(int));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*a)[i][j] = (i + j) % 10;
            (*b)[i][j] = (i - j) % 10;
            (*c)[i][j] = 0;
        }
    }
}

void free_matrices(int n, int **a, int **b, int **c) {
    for (int i = 0; i < n; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a); free(b); free(c);
}
double multiply(int n, int **a, int **b, int **c) {
    double start = omp_get_wtime();
    int j,k;

    #pragma omp parallel for private(j,k)
    for (int i = 0; i < n; i++) {
        for ( j = 0; j < n; j++) {
            int sum = 0;
            for (k = 0; k < n; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    double end = omp_get_wtime();
    return end - start;
}

int main() {
    int max_threads;
    printf("Enter number of threads : ");
    scanf("%d", &max_threads);

    printf("\n*** Analyzing Time vs. Data Size (Threads fixed at %d) ***\n", max_threads);

    int sizes[] = {512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        omp_set_num_threads(max_threads);

        int **a, **b, **c;
        init_matrices(n, &a, &b, &c);

        double time_taken = multiply(n, a, b, c);

        printf("Size: %d x %d, Threads: %d, Time: %f s\n", n, n, max_threads, time_taken);

        free_matrices(n, a, b, c);
    }
    int n = 2048; 
    printf("\n*** Analyzing Time vs Threads (Size fixed at %d x %d) ***\n", n, n);

    double base_time = 0.0;
    for (int t = 1; t <= max_threads; t *= 2) {
        omp_set_num_threads(t);

        int **a, **b, **c;
        init_matrices(n, &a, &b, &c);

        double time_taken = multiply(n, a, b, c);

        if (t == 1) {
            base_time = time_taken;
            printf("Threads: %d, Time: %f s Sequential\n", t, time_taken);
        } else {
            double speedup = base_time / time_taken;
            printf("Threads: %d, Time: %f s, Speedup: %.2fx\n", t, time_taken, speedup);
        }

        free_matrices(n, a, b, c);
    }

    return 0;
}




// #include <stdio.h>
// #include <stdlib.h>
// #include <omp.h>

// int main() {
//     int n, i, j, k,threads;
//     double s, e;

//     printf("Enter the size of the matrix: ");
//     fflush(stdout);
//     scanf("%d", &n);

//     printf("Enter the number of threads to use: ");
//     fflush(stdout);
//     scanf("%d", &threads);
//     omp_set_num_threads(threads);   

//     int **a = (int**) malloc(n * sizeof(int*));
//     int **b = (int**) malloc(n * sizeof(int*));
//     int **c = (int**) malloc(n * sizeof(int*));
//     for (i = 0; i < n; i++) {
//         a[i] = (int*) malloc(n * sizeof(int));
//         b[i] = (int*) malloc(n * sizeof(int));
//         c[i] = (int*) malloc(n * sizeof(int));
//     }

//     printf("Enter elements of Matrix A (%d x %d):\n", n, n);
//     fflush(stdout);
//     for (i = 0; i < n; i++)
//         for (j = 0; j < n; j++)
//             scanf("%d", &a[i][j]);

//     printf("Enter elements of Matrix B (%d x %d):\n", n, n);
//     fflush(stdout);
//     for (i = 0; i < n; i++)
//         for (j = 0; j < n; j++)
//             scanf("%d", &b[i][j]);

//     for (i = 0; i < n; i++)
//         for (j = 0; j < n; j++)
//             c[i][j] = 0;

//     s = omp_get_wtime();

//     #pragma omp parallel for private(j,k)
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++) {
//             int sum = 0;
//             for (k = 0; k < n; k++)
//                 sum += a[i][k] * b[k][j];
//             c[i][j] = sum;
//         }
//     }

//     e = omp_get_wtime();

//     printf("\nFinal Matrix C (A x B):\n");
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++)
//             printf("%d ", c[i][j]);
//         printf("\n");
//     }

//     printf("\nExecution Time: %f seconds\n", e - s);

//     for (i = 0; i < n; i++) {
//         free(a[i]);
//         free(b[i]);
//         free(c[i]);
//     }
//     free(a); free(b); free(c);

//     return 0;
// }

