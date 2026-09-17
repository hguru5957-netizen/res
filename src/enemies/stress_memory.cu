#include "common.cuh"

// Accurate nanosecond timer for NVIDIA GPU
// ---------------------------------------------------------
// MEMORY ENEMY KERNEL
// ---------------------------------------------------------
__global__ void memoryKernel_Write(unsigned int *k_data, int num_lines, unsigned long long run_time) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    
    unsigned long long clock_begin = gclock64();
    
    // Loop until the global timer exceeds the requested nanoseconds
    while ((gclock64() - clock_begin) < run_time) {
        // Thrash the cache by touching every 128-byte line exactly once per sweep
        for (int i = tid; i < num_lines; i += stride) {
            k_data[i * 32] = 7; 
        }
    }
}

// ---------------------------------------------------------
// MEMORY ENEMY LAUNCHER
// ---------------------------------------------------------
// allocated_sms defaults to 8 (1 GPC on Orin), but can be adjusted 
// if your partition size changes.
void launch_stress_memory(cudaStream_t stream, unsigned int* device_array, int bytesize, int stride_bytes, unsigned long long run_time, int allocated_sms) {
    int num_lines = bytesize / stride_bytes; 
    
    // Launch 2 full blocks per allocated SM to guarantee even WDU distribution
    dim3 blocks(allocated_sms * 2);
    // MAXIMIZE: 1024 threads (32 warps) per block to maximize memory requests
    dim3 threads(1024); 
    
    memoryKernel_Write<<<blocks, threads, 0, stream>>>(device_array, num_lines, run_time);
}