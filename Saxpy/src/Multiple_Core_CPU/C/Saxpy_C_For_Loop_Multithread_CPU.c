#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void saxpy(float* x, float* y, float a, long long size) {
    long long i;
    #pragma omp parallel for default(none) private(i) shared(a, x, y, size)
    for (i = 0; i < size; i++) {
        y[i] = a * x[i] + y[i];
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main() {

    int n = 30;
    int iterations = 20;
    float a = 3.1415f;
    omp_set_num_threads(16);

    double timings[n][iterations];

    for (int elements = 1; elements <= n; elements++) {

        long long size = 1 << elements;

        for (int it = 0; it < iterations; it++) {

            float* x = (float*)malloc(size * sizeof(float));
            float* y = (float*)malloc(size * sizeof(float));
            //float* z = (float*)malloc(size * sizeof(float));

            srand(time(NULL));

            for (long long i = 0; i < size; i++) {
                x[i] = ((float)rand()/(float)(RAND_MAX/10));
                y[i] = ((float)rand()/(float)(RAND_MAX/10));
            }

            double start_time = get_time();
            saxpy(x, y, a, size);
            double end_time = get_time();


            double elapsed= end_time - start_time;
            //printf("%.15lf ", elapsed);

            //double elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000000;
            timings[elements-1][it] = elapsed;

            free(x);
            free(y);
            //free(z);
        }
    }

    double Average_Time[n];

    for (int i = 0; i < n; i++) {
        Average_Time[i] = 0;
    }

    for (int z = 0; z < n; z++) {
        printf("z = %d: ", z+1);
        for (int it = 0; it < iterations; it++) {
            Average_Time[z] += timings[z][it];
            printf("%.15lf ", timings[z][it]);
        }
        printf("\n");
    }

    for (int z = 0; z < n; z++) {
        printf("Average time for size %d: %.15lf seconds\n", z, Average_Time[z]/iterations);
    }

    FILE *outfile = fopen("Saxpy/Benchmark_Results/Saxpy_C_For_Loop_Multithread_CPU16.csv", "w");

    for (int z = 0; z < n; z++) {
        fprintf(outfile, "%.15lf ", Average_Time[z]/iterations);
    }

    fclose(outfile);

    return 0;
}
