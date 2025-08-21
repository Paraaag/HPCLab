#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    for(int a=0;a<4;a++){
    int N, threads;
    printf("Enter N (vector length): "); scanf("%d",&N);
    printf("Enter threads: "); scanf("%d",&threads);

    int *a = (int*) malloc(N*sizeof(int));
    int *p = (int*) malloc(N*sizeof(int));
    for (int i=0;i<N;i++) a[i] = 1 + (i%5);


    int *chunk_sum = (int*) calloc(threads, sizeof(int));

    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int T   = omp_get_num_threads();
        int chunk = (N + T - 1) / T;
        int s = tid * chunk;
        int e = (s + chunk < N) ? (s + chunk) : N;

        int run = 0;
        for (int i=s;i<e;i++) {
            run += a[i];
            p[i] = run;
        }
        chunk_sum[tid] = run;
        #pragma omp barrier

        // Phase 2: compute offsets (serial prefix on chunk_sum; small)
        int offset = 0;
        for (int k=0;k<tid;k++) offset += chunk_sum[k];

        // Phase 3: add offset to this chunk
        for (int i=s;i<e;i++) p[i] += offset;
    }
    double t1 = omp_get_wtime();

    printf("Last prefix value: %d\n", p[N-1]);
    printf("Time (parallel block-scan): %.6f s, N=%d, threads=%d\n", (t1-t0), N, threads);

    free(a); free(p); free(chunk_sum);}
    return 0;
}