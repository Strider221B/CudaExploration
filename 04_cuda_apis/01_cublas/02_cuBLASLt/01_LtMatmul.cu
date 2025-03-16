#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(status) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            cerr << "cuBLAS error at line " << __LINE__ << ": " << status << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    cout << "Matrix " << name << ":" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << setw(8) << fixed << setprecision(2) << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {

    // from cublas-lt docs - "Dimensions m and k must be multiples of 4."
    // this means we can't have a 3x4 or 2x4 matrix, but we can have a 4x4 or 4x8 matrix
    const int M = 4, K = 4, N = 4;

    // Manually define input matrices
    float h_A[M * K] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    float h_B[K * N] = {
        1.0f, 2.0f, 4.0f, 4.0f,     // changed the 3.0f to 4.0f
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        17.0f, 18.0f, 19.0f, 20.0f  // changed the last row to 17.0f, 18.0f, 19.0f, 20.0f
    };
    // we remember to be careful by not making A and B the same

    float h_C_cpu[M * N] = {0};
    float h_C_gpu_fp32[M * N] = {0};
    float h_C_gpu_fp16[M * N] = {0};

    // Print input matrices
    print_matrix(h_A, M, K, "A");
    print_matrix(h_B, K, N, "B");

    // Allocate device memory for FP32
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));

    // Allocate device memory for FP16
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(half)));

    // Copy data to device (FP32)
    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Convert and copy data to device (FP16)
    vector<half> h_A_half(M * K);
    vector<half> h_B_half(K * N);
    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Set up matrix descriptors for FP32
    cublasLtMatrixLayout_t matA_fp32, matB_fp32, matC_fp32;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp32, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp32, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp32, CUDA_R_32F, N, M, N));

    // Set up matrix descriptors for FP16
    cublasLtMatrixLayout_t matA_fp16, matB_fp16, matC_fp16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp16, CUDA_R_16F, K, M, K)); // original MKK
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp16, CUDA_R_16F, N, K, N)); // original KNN
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp16, CUDA_R_16F, N, M, N)); // original MNN

    // Set up matrix multiplication descriptor for FP32
    cublasLtMatmulDesc_t matmulDesc_fp32;
    // Notes: 
    // CUBLAS_COMPUTE_32F refers to the precision of intermediate operations during the computation process.
    // CUDA_R_32F refers to the data type of the actual matrix elements in memory.
    // Why the Difference Matters - For advanced use cases, such as mixed precision training (common in deep learning), you might have:
    // Matrices stored in lower precision (e.g., CUDA_R_16F for half-precision). 
    // Computations performed in higher precision (e.g., CUBLAS_COMPUTE_32F) to maintain numerical stability.
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set up matrix multiplication descriptor for FP16
    cublasLtMatmulDesc_t matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    // Set matrix operation for A and B
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

    // Set up alpha and beta
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication using cublasLtMatmul (FP32)
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc_fp32, &alpha, d_B_fp32, matB_fp32, d_A_fp32, matA_fp32, &beta, d_C_fp32, matC_fp32, d_C_fp32, matC_fp32, nullptr, nullptr, 0, 0));

    // half alpha and beta
    const half alpha_half = __float2half(1.0f);
    const half beta_half = __float2half(0.0f);
    
    // Perform matrix multiplication using cublasLtMatmul (FP16)
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc_fp16, &alpha_half, d_B_fp16, matB_fp16, d_A_fp16, matA_fp16, &beta_half, d_C_fp16, matC_fp16, d_C_fp16, matC_fp16, nullptr, nullptr, 0, 0));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    vector<half> h_C_gpu_fp16_half(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp16_half.data(), d_C_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Convert half precision results to single precision
    for (int i = 0; i < M * N; ++i) {
        h_C_gpu_fp16[i] = __half2float(h_C_gpu_fp16_half[i]);
    }

    // Perform CPU matrix multiplication
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);

    // Print results
    print_matrix(h_C_cpu, M, N, "C (CPU)");
    print_matrix(h_C_gpu_fp32, M, N, "C (GPU FP32)");
    print_matrix(h_C_gpu_fp16, M, N, "C (GPU FP16)");

    // Compare CPU and GPU results
    bool fp32_match = true;
    bool fp16_match = true;
    for (int i = 0; i < M * N; ++i) {
        if (abs(h_C_cpu[i] - h_C_gpu_fp32[i]) > 1e-5) {
            fp32_match = false;
        }
        if (abs(h_C_cpu[i] - h_C_gpu_fp16[i]) > 1e-2) {  // Increased tolerance for FP16
            fp16_match = false;
        }
    }

    cout << "FP32 Results " << (fp32_match ? "match" : "do not match") << endl;
    cout << "FP16 Results " << (fp16_match ? "match" : "do not match") << endl;

    // Clean up
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp16));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp32));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32));
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));

    return 0;
}

// Output:
// ~/Git/CudaExploration/04_cuda_apis/01_cublas/02_cuBLASLt$ nvcc -lcublasLt -o 01_exec 01_LtMatmul.cu 
// ~/Git/CudaExploration/04_cuda_apis/01_cublas/02_cuBLASLt$ ./01_exec 
// Matrix A:
//     1.00     2.00     3.00     4.00 
//     5.00     6.00     7.00     8.00 
//     9.00    10.00    11.00    12.00 
//    13.00    14.00    15.00    16.00 

// Matrix B:
//     1.00     2.00     4.00     4.00 
//     5.00     6.00     7.00     8.00 
//     9.00    10.00    11.00    12.00 
//    17.00    18.00    19.00    20.00 

// Matrix C (CPU):
//   106.00   116.00   127.00   136.00 
//   234.00   260.00   291.00   312.00 
//   362.00   404.00   455.00   488.00 
//   490.00   548.00   619.00   664.00 

// Matrix C (GPU FP32):
//   106.00   116.00   127.00   136.00 
//   234.00   260.00   291.00   312.00 
//   362.00   404.00   455.00   488.00 
//   490.00   548.00   619.00   664.00 

// Matrix C (GPU FP16):
//   106.00   116.00   127.00   136.00 
//   234.00   260.00   291.00   312.00 
//   362.00   404.00   455.00   488.00 
//   490.00   548.00   619.00   664.00 

// FP32 Results match
// FP16 Results match
