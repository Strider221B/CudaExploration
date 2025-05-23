#include <cublasXt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

const int M = 1024 / 4;
const int N = 1024 / 4;
const int K = 1024 / 4;

#define CHECK_CUBLAS(call) { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "Error in " << #call << ", line " << __LINE__ << std::endl; exit(1); } }

// Observe that there are no cuda memcopy in this code. cuBlasXt handles it for us but at an performance overhead.

// cuBLAS-Xt is designed for high-performance computing and is optimized for large-scale matrix operations. 
// Its architecture is based on a tile-based approach, where matrices are divided into smaller tiles that are processed parallel. 
// This approach allows for efficient use of the GPU's memory hierarchy and enables high-bandwidth data transfer between the GPU and the host system.

int main() {
    // Initialize random number generator
    srand(time(0));

    // Allocate host memory for matrices
    float* A_host = new float[M * K];
    float* B_host = new float[K * N];
    float* C_host_cpu = new float[M * N];
    float* C_host_gpu = new float[M * N];

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) {
        A_host[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B_host[i] = (float)rand() / RAND_MAX;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_host_cpu[i * N + j] = 0.0f;
            for (int k = 0; k < K; k++) {
                C_host_cpu[i * N + j] += A_host[i * K + k] * B_host[k * N + j];
            }
        }
    }

    cublasXtHandle_t handle;
    CHECK_CUBLAS(cublasXtCreate(&handle));


    int devices[1] = {0};
    CHECK_CUBLAS(cublasXtDeviceSelect(handle, 1, devices));

    // Warmup run
    CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_host, N, A_host, K, &beta, C_host_gpu, N));

    float max_diff = 1e-4f;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(C_host_cpu[i] - C_host_gpu[i]);
        if (diff > max_diff) {
            std::cout << "i: " << i << " CPU: " << C_host_cpu[i] << ", GPU: " << C_host_gpu[i] << std::endl;
            
        }
    }

    std::cout << "Maximum difference between CPU and GPU results: " << max_diff << std::endl;

    // Free memory
    delete[] A_host;
    delete[] B_host;
    delete[] C_host_cpu;
    delete[] C_host_gpu;


    return 0;

}

// Output:
// ~/Git/CudaExploration/04_cuda_apis/01_cublas/03_cuBLASXt$ nvcc -lcublas -o ./01_exec ./01_demo.cu 
// ~/Git/CudaExploration/04_cuda_apis/01_cublas/03_cuBLASXt$ ./01_exec 
// Maximum difference between CPU and GPU results: 0.0001
