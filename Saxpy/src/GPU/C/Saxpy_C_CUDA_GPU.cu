#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


__global__ void saxpy(float *x, float *y, float a, long long size )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        y[i] = a * x[i] + y[i];
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main() {

    int n = 29;
    int iterations = 20;
    float a = 3.1415f;

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


            //printf("%f\n", x[0]);
            //printf("\n");
            //printf("%f\n", y[0]);
            //printf("\n");

            float* d_x, *d_y;
            cudaMalloc(&d_x, size * sizeof(float));
            cudaMalloc(&d_y, size * sizeof(float));

            cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice);

            int threadsPerBlock = 1024;
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


            double start_time = get_time();
            saxpy<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, a, size);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            
            cudaDeviceSynchronize();
            double end_time = get_time();

            cudaMemcpy(y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost);

           
            //printf("%f\n", y[0]);
            //printf("\n");


            double elapsed= end_time - start_time;
            //printf("%.15lf ", elapsed);

            //double elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000000;
            timings[elements-1][it] = elapsed;

            free(x);
            free(y);
            cudaFree(d_x);
            cudaFree(d_y);
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

    FILE *outfile = fopen("Saxpy/Benchmark_Results/Saxpy_C_CUDA_GPU.csv", "w");

    for (int z = 0; z < n; z++) {
        fprintf(outfile, "%.15lf ", Average_Time[z]/iterations);
    }

    fclose(outfile);

    return 0;
}
