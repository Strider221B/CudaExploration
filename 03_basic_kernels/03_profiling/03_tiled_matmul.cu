#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
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
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplyOptimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

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
// ~/Git/CudaExploration/03_basic_kernels/03_profiling$ nvcc -o 03_exec 03_tiled_matmul.cu -lnvToolsExt
// ~/Git/CudaExploration/03_basic_kernels/03_profiling$ nsys profile --stats=true ./03_exec
// Generating '/tmp/nsys-report-eeab.qdstrm'
// [1/8] [========================100%] report2.nsys-rep
// [2/8] [========================100%] report2.sqlite
// [3/8] Executing 'nvtx_sum' stats report
// SKIPPED: /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.sqlite does not contain NV Tools Extension (NVTX) data.
// [4/8] Executing 'osrt_sum' stats report

//  Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)    Min (ns)  Max (ns)   StdDev (ns)       Name     
//  --------  ---------------  ---------  ----------  -----------  --------  ---------  -----------  --------------
//      78.3        702378854         11  63852623.1  100147321.0    153365  100355173   44776797.3  poll          
//      17.7        158574518        460    344727.2      81409.0      1005   14530923    1158969.6  ioctl         
//       3.0         26939996          6   4489999.3    2179004.5      2513   11566423    5244586.8  fread         
//       0.3          2853838          7    407691.1     650948.0      1307     803810     382591.3  read          
//       0.3          2590832          3    863610.7    1186720.0     99705    1304407     664173.5  pthread_create
//       0.2          2170041         20    108502.1       8492.5      1206     672354     220612.4  fopen         
//       0.1           457789         10     45778.9       2262.0      1005     367634     114020.9  fclose        
//       0.0           221003          1    221003.0     221003.0    221003     221003          0.0  pthread_join  
//       0.0           182221          2     91110.5      91110.5     85130      97091       8457.7  sem_timedwait 
//       0.0           150665         24      6277.7       5679.0      1307      11458       2979.1  mmap          
//       0.0           115483          4     28870.8      28594.0     27640      30655       1324.4  write         
//       0.0            78191          8      9773.9       9949.5      4021      21206       5518.2  open          
//       0.0            47943          5      9588.6       2714.0      1508      36887      15290.8  fwrite        
//       0.0            28543          1     28543.0      28543.0     28543      28543          0.0  fgets         
//       0.0            11256          6      1876.0       1809.0      1005       2914        822.6  fcntl         
//       0.0            10654          3      3551.3       4322.0      1408       4924       1880.4  pipe2         
//       0.0             2312          1      2312.0       2312.0      2312       2312          0.0  fflush        

// [5/8] Executing 'cuda_api_sum' stats report

//  Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Name         
//  --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------
//      74.7        555558995          1  555558995.0  555558995.0  555558995  555558995          0.0  cudaLaunchKernel      
//      24.3        180346510          3   60115503.3     487165.0     375701  179483644  103375857.2  cudaMalloc            
//       1.0          7067158          1    7067158.0    7067158.0    7067158    7067158          0.0  cudaDeviceSynchronize 
//       0.1           655718          3     218572.7     192272.0     171368     292078      64509.8  cudaFree              
//       0.0             1307          1       1307.0       1307.0       1307       1307          0.0  cuModuleGetLoadingMode

// [6/8] Executing 'cuda_gpu_kern_sum' stats report

//  Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                Name                               
//  --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------------------------------------------------------------
//     100.0          7085530          1  7085530.0  7085530.0   7085530   7085530          0.0  matrixMultiplyOptimized(float *, float *, float *, int, int, int)

// [7/8] Executing 'cuda_gpu_mem_time_sum' stats report
// SKIPPED: /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.sqlite does not contain GPU memory data.
// [8/8] Executing 'cuda_gpu_mem_size_sum' stats report
// SKIPPED: /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.sqlite does not contain GPU memory data.
// Generated:
//     /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.nsys-rep
//     /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.sqlite

// Memory throughput: 41.03 Gbytes/s
