#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
int main() {

    // Define matrix dimensions
    const int M = 1024; // Number of rows in A and C
    const int N = 1024; // Number of columns in B and C
    const int K = 1024; // Number of columns in A and rows in B

    // Calculate matrix sizes in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);


    // Kernel launch code
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Synchronize device
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;

}

// Output
// somesh@LAPTOP-OV0CBJU0:~/Git/CudaExploration/03_basic_kernels/03_profiling$ nvcc -o 02_exec 02_naive_matmul.cu -lnvToolsExt
// somesh@LAPTOP-OV0CBJU0:~/Git/CudaExploration/03_basic_kernels/03_profiling$ nsys profile --stats=true ./02_exec
// Generating '/tmp/nsys-report-a1b4.qdstrm'
// [1/8] [========================100%] report1.nsys-rep
// [2/8] [========================100%] report1.sqlite
// [3/8] Executing 'nvtx_sum' stats report
// SKIPPED: /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report1.sqlite does not contain NV Tools Extension (NVTX) data.
// [4/8] Executing 'osrt_sum' stats report

//  Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)    Min (ns)  Max (ns)   StdDev (ns)       Name     
//  --------  ---------------  ---------  ----------  -----------  --------  ---------  -----------  --------------
//      78.2        702010245         11  63819113.2  100144534.0    147833  100242423   45939752.6  poll          
//      17.9        160489131        449    357436.8      70464.0      1000   14660527    1201744.7  ioctl         
//       2.9         25574547          6   4262424.5    2029795.0      3002   11358525    5124757.2  fread         
//       0.4          3478745          3   1159581.7    1540892.0    287562    1650291     757169.6  pthread_create
//       0.3          2638585          8    329823.1     313433.0      1301     688422     351379.7  read          
//       0.2          1756287         19     92436.2       8207.0      1000     644181     183649.5  fopen         
//       0.0           307681         24     12820.0       6957.0      2302      73267      16249.8  mmap          
//       0.0           302275         10     30227.5       2302.5      1101     213593      66058.5  fclose        
//       0.0           255732          3     85244.0      83075.0     80073      92584       6531.4  sem_timedwait 
//       0.0           172756          1    172756.0     172756.0    172756     172756          0.0  pthread_join  
//       0.0           117608          4     29402.0      29527.0     28026      30528       1267.0  write         
//       0.0            55951          4     13987.8       2452.5      1001      50045      24052.1  fcntl         
//       0.0            39735          7      5676.4       6005.0      3102       8008       1947.6  open          
//       0.0            26323          1     26323.0      26323.0     26323      26323          0.0  fgets         
//       0.0            18015          4      4503.8       2502.0      1501      11510       4701.0  fwrite        
//       0.0            12912          3      4304.0       5205.0      1301       6406       2669.1  pipe2         
//       0.0             2102          1      2102.0       2102.0      2102       2102          0.0  fflush        

// [5/8] Executing 'cuda_api_sum' stats report

//  Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Name         
//  --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------
//      70.7        540738346          1  540738346.0  540738346.0  540738346  540738346          0.0  cudaLaunchKernel      
//      27.6        211510369          3   70503456.3     720355.0     621565  210168449  120953441.8  cudaMalloc            
//       1.5         11496751          1   11496751.0   11496751.0   11496751   11496751          0.0  cudaDeviceSynchronize 
//       0.2          1376051          3     458683.7     429290.0     421884     524877      57444.6  cudaFree              
//       0.0              800          1        800.0        800.0        800        800          0.0  cuModuleGetLoadingMode

// [6/8] Executing 'cuda_gpu_kern_sum' stats report

//  Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                            Name                          
//  --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  --------------------------------------------------------
//     100.0         11431056          1  11431056.0  11431056.0  11431056  11431056          0.0  matrixMultiply(float *, float *, float *, int, int, int)

// [7/8] Executing 'cuda_gpu_mem_time_sum' stats report
// SKIPPED: /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report1.sqlite does not contain GPU memory data.
// [8/8] Executing 'cuda_gpu_mem_size_sum' stats report
// SKIPPED: /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report1.sqlite does not contain GPU memory data.
// Generated:
//     /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report1.nsys-rep
//     /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report1.sqlite

// Memory throughput: 25.33 Gbyte/s