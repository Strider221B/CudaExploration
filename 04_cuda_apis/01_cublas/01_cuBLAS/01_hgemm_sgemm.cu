// dedicated for small handwritten matrices
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define M 3
#define K 4
#define N 2

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float B[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float C_cpu[M * N], C_cublas_s[M * N], C_cublas_h[M * N];

    // CPU matmul
    cpu_matmul(A, B, C_cpu);

    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // row major A = 
    // 1.0 2.0 3.0 4.0
    // 5.0 6.0 7.0 8.0

    // col major A = 
    // 1.0 5.0
    // 2.0 6.0
    // 3.0 7.0
    // 4.0 8.0

    // memory layout (row)
    // 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

    // memory layout (col)
    // 1.0 5.0 2.0 6.0 3.0 7.0 4.0 8.0
    
    // cuBLAS SGEMM
    // cublasSgemm performs C = alpha*operation(A) . operation(B) + beta*C. We have given alpha as 1 and beta as 0 so for us
    // it becomes C = operation(A) . operation(B); a simple dot product.
    // Note: cublasSgemm expects the matrix in column major format and operates it that way. In order to avoid additional handling on 
    // on our sidee we need to pass the parameters bit different - matrix dimension being passed as N M K instead of M N K.
    // Refer: https://stackoverflow.com/a/56064726 
    // Gist:  So instead of computing AB = C, we do [B^T A^T] = [C^T] .Luckily, [B^T] and [A^T] we already obtained
    // by the very action of creating A and B in row-major order, so we can simply bypass the transposition with CUBLAS_OP_N
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cuBLAS HGEMM
    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    // Convert to half precision on CPU
    half A_h[M * K], B_h[K * N];
    for (int i = 0; i < M * K; i++) {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = __float2half(B[i]);
    }

    // Copy half precision data to device
    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));

    // Copy result back to host and convert to float
    half C_h[M * N];
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    // Print results
    printf("Matrix A (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);
    printf("Matrix B (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);
    printf("CPU Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cpu, M, N);
    printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_s, M, N);
    printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

// Output:
// Note you need to add -cublas to nvcc to link cublas. It won't come in by default.
// ~/Git/CudaExploration/04_cuda_apis/01_cublas/01_cuBLAS$ nvcc -lcublas -o 01_exec 01_hgemm_sgemm.cu
// ~/Git/CudaExploration/04_cuda_apis/01_cublas/01_cuBLAS$ ./01_exec 
// Matrix A (3x4):
//    1.000    2.000    3.000    4.000 
//    5.000    6.000    7.000    8.000 
//    9.000   10.000   11.000   12.000 

// Matrix B (4x2):
//    1.000    2.000 
//    3.000    4.000 
//    5.000    6.000 
//    7.000    8.000 

// CPU Result (3x2):
//   50.000   60.000 
//  114.000  140.000 
//  178.000  220.000 

// cuBLAS SGEMM Result (3x2):
//   50.000   60.000 
//  114.000  140.000 
//  178.000  220.000 

// cuBLAS HGEMM Result (3x2):
//   50.000   60.000 
//  114.000  140.000 
//  178.000  220.000 
