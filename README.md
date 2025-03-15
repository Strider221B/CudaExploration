# CudaExploration
  
## The code was run on the following GPU  
  
### nvidia-smi  
+-----------------------------------------------------------------------------------------+  
| NVIDIA-SMI 560.51                 Driver Version: 561.19         CUDA Version: 12.6     |  
|-----------------------------------------+------------------------+----------------------+  
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |  
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |  
|                                         |                        |               MIG M. |  
|=========================================+========================+======================|  
|   0  NVIDIA GeForce GTX 1650        On  |   00000000:01:00.0  On |                  N/A |  
| N/A   47C    P8              7W /   50W |     766MiB /   4096MiB |      1%      Default |  
|                                         |                        |                  N/A |  
+-----------------------------------------+------------------------+----------------------+  
  
+-----------------------------------------------------------------------------------------+  
| Processes:                                                                              |  
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |  
|        ID   ID                                                               Usage      |  
|=========================================================================================|  
|  No running processes found                                                             |  
+-----------------------------------------------------------------------------------------+  
  
### deviceQuery  
  
Detected 1 CUDA Capable device(s)  
  
Device 0: "NVIDIA GeForce GTX 1650"  
  CUDA Driver Version / Runtime Version          12.6 / 12.6  
  CUDA Capability Major/Minor version number:    7.5  
  Total amount of global memory:                 4096 MBytes (4294639616 bytes)  
  (014) Multiprocessors, (064) CUDA Cores/MP:    896 CUDA Cores  
  GPU Max Clock rate:                            1515 MHz (1.51 GHz)  
  Memory Clock rate:                             6001 Mhz  
  Memory Bus Width:                              128-bit  
  L2 Cache Size:                                 1048576 bytes  
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)  
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers  
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers  
  Total amount of constant memory:               65536 bytes  
  Total amount of shared memory per block:       49152 bytes  
  Total shared memory per multiprocessor:        65536 bytes  
  Total number of registers available per block: 65536  
  Warp size:                                     32  
  Maximum number of threads per multiprocessor:  1024  
  Maximum number of threads per block:           1024  
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)  
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)  
  Maximum memory pitch:                          2147483647 bytes  
  Texture alignment:                             512 bytes  
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)  
  Run time limit on kernels:                     Yes  
  Integrated GPU sharing Host Memory:            No  
  Support host page-locked memory mapping:       Yes  
  Alignment requirement for Surfaces:            Yes  
  Device has ECC support:                        Disabled  
  Device supports Unified Addressing (UVA):      Yes  
  Device supports Managed Memory:                Yes  
  Device supports Compute Preemption:            Yes  
  Supports Cooperative Kernel Launch:            Yes  
  Supports MultiDevice Co-op Kernel Launch:      No  
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0  
  Compute Mode:  
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >  
  
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.6, CUDA Runtime Version = 12.6, NumDevs = 1  
  
### CPU Information  
  
Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz   2.50 GHz  

### RAM  
  
16.0 GB (15.8 GB usable)  