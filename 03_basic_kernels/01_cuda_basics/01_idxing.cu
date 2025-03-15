#include <stdio.h>

__global__ void whoami(void) {
    int block_id =
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
    // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so 
    // we will get 2 warp of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}

// Output:
// Note: What's interesting here is - this is last 64 entries in the output. You'll notice that it has 2 blocks - grouped together:
// 32 entries for Block(1 0 3) and 32 for Block(0 1 3). That matches the warp size of the GPU I am currently using, which is 32.
// So, the block order isn't guaranteed. Nor is it guaranteed that all threads in a block will be executed in a particular fashion.
// What is guaranteed is that WARP number of elements within the block will be executed together; and in a very specigic order.
// If the thread index is referenced as Thread(x y z); then first x is incremented till hits the max allowed number; then y and then 
// finally z.  

// Last 64 entries:
// 1248 | Block(1 0 3) =  19 | Thread(0 0 2) =  32
// 1249 | Block(1 0 3) =  19 | Thread(1 0 2) =  33
// 1250 | Block(1 0 3) =  19 | Thread(2 0 2) =  34
// 1251 | Block(1 0 3) =  19 | Thread(3 0 2) =  35
// 1252 | Block(1 0 3) =  19 | Thread(0 1 2) =  36
// 1253 | Block(1 0 3) =  19 | Thread(1 1 2) =  37
// 1254 | Block(1 0 3) =  19 | Thread(2 1 2) =  38
// 1255 | Block(1 0 3) =  19 | Thread(3 1 2) =  39
// 1256 | Block(1 0 3) =  19 | Thread(0 2 2) =  40
// 1257 | Block(1 0 3) =  19 | Thread(1 2 2) =  41
// 1258 | Block(1 0 3) =  19 | Thread(2 2 2) =  42
// 1259 | Block(1 0 3) =  19 | Thread(3 2 2) =  43
// 1260 | Block(1 0 3) =  19 | Thread(0 3 2) =  44
// 1261 | Block(1 0 3) =  19 | Thread(1 3 2) =  45
// 1262 | Block(1 0 3) =  19 | Thread(2 3 2) =  46
// 1263 | Block(1 0 3) =  19 | Thread(3 3 2) =  47
// 1264 | Block(1 0 3) =  19 | Thread(0 0 3) =  48
// 1265 | Block(1 0 3) =  19 | Thread(1 0 3) =  49
// 1266 | Block(1 0 3) =  19 | Thread(2 0 3) =  50
// 1267 | Block(1 0 3) =  19 | Thread(3 0 3) =  51
// 1268 | Block(1 0 3) =  19 | Thread(0 1 3) =  52
// 1269 | Block(1 0 3) =  19 | Thread(1 1 3) =  53
// 1270 | Block(1 0 3) =  19 | Thread(2 1 3) =  54
// 1271 | Block(1 0 3) =  19 | Thread(3 1 3) =  55
// 1272 | Block(1 0 3) =  19 | Thread(0 2 3) =  56
// 1273 | Block(1 0 3) =  19 | Thread(1 2 3) =  57
// 1274 | Block(1 0 3) =  19 | Thread(2 2 3) =  58
// 1275 | Block(1 0 3) =  19 | Thread(3 2 3) =  59
// 1276 | Block(1 0 3) =  19 | Thread(0 3 3) =  60
// 1277 | Block(1 0 3) =  19 | Thread(1 3 3) =  61
// 1278 | Block(1 0 3) =  19 | Thread(2 3 3) =  62
// 1279 | Block(1 0 3) =  19 | Thread(3 3 3) =  63
// 1312 | Block(0 1 3) =  20 | Thread(0 0 2) =  32
// 1313 | Block(0 1 3) =  20 | Thread(1 0 2) =  33
// 1314 | Block(0 1 3) =  20 | Thread(2 0 2) =  34
// 1315 | Block(0 1 3) =  20 | Thread(3 0 2) =  35
// 1316 | Block(0 1 3) =  20 | Thread(0 1 2) =  36
// 1317 | Block(0 1 3) =  20 | Thread(1 1 2) =  37
// 1318 | Block(0 1 3) =  20 | Thread(2 1 2) =  38
// 1319 | Block(0 1 3) =  20 | Thread(3 1 2) =  39
// 1320 | Block(0 1 3) =  20 | Thread(0 2 2) =  40
// 1321 | Block(0 1 3) =  20 | Thread(1 2 2) =  41
// 1322 | Block(0 1 3) =  20 | Thread(2 2 2) =  42
// 1323 | Block(0 1 3) =  20 | Thread(3 2 2) =  43
// 1324 | Block(0 1 3) =  20 | Thread(0 3 2) =  44
// 1325 | Block(0 1 3) =  20 | Thread(1 3 2) =  45
// 1326 | Block(0 1 3) =  20 | Thread(2 3 2) =  46
// 1327 | Block(0 1 3) =  20 | Thread(3 3 2) =  47
// 1328 | Block(0 1 3) =  20 | Thread(0 0 3) =  48
// 1329 | Block(0 1 3) =  20 | Thread(1 0 3) =  49
// 1330 | Block(0 1 3) =  20 | Thread(2 0 3) =  50
// 1331 | Block(0 1 3) =  20 | Thread(3 0 3) =  51
// 1332 | Block(0 1 3) =  20 | Thread(0 1 3) =  52
// 1333 | Block(0 1 3) =  20 | Thread(1 1 3) =  53
// 1334 | Block(0 1 3) =  20 | Thread(2 1 3) =  54
// 1335 | Block(0 1 3) =  20 | Thread(3 1 3) =  55
// 1336 | Block(0 1 3) =  20 | Thread(0 2 3) =  56
// 1337 | Block(0 1 3) =  20 | Thread(1 2 3) =  57
// 1338 | Block(0 1 3) =  20 | Thread(2 2 3) =  58
// 1339 | Block(0 1 3) =  20 | Thread(3 2 3) =  59
// 1340 | Block(0 1 3) =  20 | Thread(0 3 3) =  60
// 1341 | Block(0 1 3) =  20 | Thread(1 3 3) =  61
// 1342 | Block(0 1 3) =  20 | Thread(2 3 3) =  62
// 1343 | Block(0 1 3) =  20 | Thread(3 3 3) =  63
