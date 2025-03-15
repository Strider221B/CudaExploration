#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
// #define N 300  // Number of columns in B and C

#define BLOCK_SIZE 32

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)

void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; //Optimized
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int col = blockIdx.y * blockDim.y + threadIdx.y; //Not optimized but correct
    // int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // The second way of declaring dim3 has no impact it seems because our final result is a symmetric 
    // matrix. M and N are both 256. However changing N to 300 will clearly show that the second option
    // is incorrect. This is because inside the kernel, we are using col of result as blockIdx.x; 
    // so blockIdx.x max value should correspond to the number of cols in result which is N.
    // For this to work, the kernel code needs to be changed to:
    //     int row = blockIdx.x * blockDim.x + threadIdx.y;
    //     int col = blockIdx.y * blockDim.y + threadIdx.x;

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-4) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

//Output: 
// Optimized, i.e.
//     int row = blockIdx.y * blockDim.y + threadIdx.y; //Optimized
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
// Performing warm-up runs...
// Benchmarking CPU implementation...
// Benchmarking GPU implementation...
// Results are correct
// CPU average time: 125058.136500 microseconds
// GPU average time: 347.683500 microseconds
// Speedup: 359.689593x

// Not optimized, i.e.:
//     int col = blockIdx.y * blockDim.y + threadIdx.y; //Not optimized but correct
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
// Performing warm-up runs...
// Benchmarking CPU implementation...
// Benchmarking GPU implementation...
// Results are correct
// CPU average time: 128374.351000 microseconds
// GPU average time: 3707.629100 microseconds
// Speedup: 34.624378x

// The reason why the optimized version is so fast is the way WARP schedules these threads.
// In the optimized version col is incremented as threadIdx.x while in not optimized version
// row is incremented as threadIdx.x. Now data is stored such that all the column values are stored 
// one after another; so when we do B[l * n + col]; we are accessing contiguous "global" memory location.
// This is way faster if all the threads in a WARP can access contiguous memory location as the memory lookups
// are slow and the memory calls can be grouped together. In the other approach the non contiguous memory locations
// are getting accessed so it's not optimized; hence slower.