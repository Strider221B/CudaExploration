#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMul(float* A, float* B, float* C, int N) {
    nvtxRangePush("Matrix Multiplication");
    
    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();  // End of Matrix Multiplication
}

int main() {
    const int N = 1024;
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    // Initialize matrices A and B here...

    matrixMul(A, B, C, N);

    // Use result in C...

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

// Output:
//
//On WSL we don't get the complete report by default because of a limitation in CUDA toolkit:
// https://forums.developer.nvidia.com/t/nsys-is-not-collecting-kernel-data/244647/33
// Workaround:
// Run these 2 lines on the terminal:
// mkdir -p "$(dirname "$(nsys -z)")"
// echo 'CuptiUseRawGpuTimestamps=false' >> "$(nsys -z)"
//
// Output from code:
// ~/Git/CudaExploration/03_basic_kernels/03_profiling$ nvcc -o 01_exec 01_nvtx_matmul.cu -lnvToolsExt
// ~/Git/CudaExploration/03_basic_kernels/03_profiling$ nsys profile --stats=true ./01_exec 
// Generating '/tmp/nsys-report-09e9.qdstrm'
// [1/8] [========================100%] report2.nsys-rep
// [2/8] [========================100%] report2.sqlite
// [3/8] Executing 'nvtx_sum' stats report

//  Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style           Range         
//  --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  ----------------------
//      50.0        608636630          1  608636630.0  608636630.0  608636630  608636630          0.0  PushPop  :Matrix Multiplication
//      47.5        578162733          1  578162733.0  578162733.0  578162733  578162733          0.0  PushPop  :Memory Allocation    
//       1.3         15985298          1   15985298.0   15985298.0   15985298   15985298          0.0  PushPop  :Kernel Execution     
//       0.7          8779399          1    8779399.0    8779399.0    8779399    8779399          0.0  PushPop  :Memory Copy H2D      
//       0.3          4207900          1    4207900.0    4207900.0    4207900    4207900          0.0  PushPop  :Memory Copy D2H      
//       0.1          1496500          1    1496500.0    1496500.0    1496500    1496500          0.0  PushPop  :Memory Deallocation  

// [4/8] Executing 'osrt_sum' stats report

//  Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)       Name     
//  --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  --------------
//      55.0        199944108          6  33324018.0  29936151.5    140200  87464805   34437864.8  poll          
//      36.7        133408907        461    289390.3     76400.0      1000  12138901     883282.9  ioctl         
//       5.7         20814501          7   2973500.1    500000.0      2100  10257301    4476613.9  fread         
//       0.7          2595300          8    324412.5    307550.0      1100    680300     345842.5  read          
//       0.7          2577700          3    859233.3   1168900.0    103300   1305500     658210.7  pthread_create
//       0.4          1555501          3    518500.3     92600.0     82900   1380001     746097.2  sem_timedwait 
//       0.3           931100         15     62073.3      5600.0      1200    479100     149121.4  fopen         
//       0.2           695500          7     99357.1      6600.0      3000    660000     247224.6  open          
//       0.1           266200          8     33275.0      2150.0      1400    185200      63388.3  fclose        
//       0.1           229100          1    229100.0    229100.0    229100    229100          0.0  pthread_join  
//       0.0           162200         24      6758.3      6350.0      1600     13100       3263.2  mmap          
//       0.0           120300          4     30075.0     30550.0     28000     31200       1486.3  write         
//       0.0            36200          2     18100.0     18100.0     11200     25000       9758.1  fgets         
//       0.0            11000          3      3666.7      3900.0      1200      5900       2358.7  pipe2         
//       0.0             4300          2      2150.0      2150.0      1600      2700        777.8  fwrite        
//       0.0             4100          2      2050.0      2050.0      1400      2700        919.2  fcntl         

// [5/8] Executing 'cuda_api_sum' stats report

//  Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name         
//  --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------
//      85.6        181093511          3  60364503.7    783200.0    647100  179663211  103315733.6  cudaMalloc            
//       6.1         12975999          3   4325333.0   4205400.0    873500    7897099    3513335.1  cudaMemcpy            
//       5.4         11405199          1  11405199.0  11405199.0  11405199   11405199          0.0  cudaDeviceSynchronize 
//       2.2          4576099          1   4576099.0   4576099.0   4576099    4576099          0.0  cudaLaunchKernel      
//       0.7          1490600          3    496866.7    472300.0    451700     566600      61262.9  cudaFree              
//       0.0             1300          1      1300.0      1300.0      1300       1300          0.0  cuModuleGetLoadingMode

// [6/8] Executing 'cuda_gpu_kern_sum' stats report

//  Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                       Name                      
//  --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  -----------------------------------------------
//     100.0         11369218          1  11369218.0  11369218.0  11369218  11369218          0.0  matrixMulKernel(float *, float *, float *, int)

// [7/8] Executing 'cuda_gpu_mem_time_sum' stats report

//  Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
//  --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
//      70.1          3344992      1  3344992.0  3344992.0   3344992   3344992          0.0  [CUDA memcpy Device-to-Host]
//      29.9          1425371      2   712685.5   712685.5    703006    722365      13688.9  [CUDA memcpy Host-to-Device]

// [8/8] Executing 'cuda_gpu_mem_size_sum' stats report

//  Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
//  ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
//       8.389      2     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]
//       4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]

// Generated:
//     /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.nsys-rep
//     /home/somesh/Git/CudaExploration/03_basic_kernels/03_profiling/report2.sqlite
//
// Launch UI profiler -> /usr/local$ ./cuda-12.6/nsight-compute-2024.3.0/ncu-ui
// file to open: report2.nsys-rep
// for profiling via PM Sampling, ensure you launch NVidia Control Panel with admin rights and then enable developer options and then select
// "Allow GPU Performance Counters to all users."
// Also, when profiling ensure detailed metrics is enabled as we are interested in memory workload -> memory throughput. 
// Higher the throughput, better it is.
