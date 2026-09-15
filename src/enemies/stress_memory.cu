#include "common.cuh"

__global__ void memoryKernel_Write(unsigned int *k_data, int num_lines, unsigned long long run_time) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    
    unsigned long long clock_begin = clock64();
    
    while ((clock64() - clock_begin) < run_time) {
        // Thrash the cache by touching every 128-byte line exactly once per sweep
        for (int i = tid; i < num_lines; i += stride) {
            k_data[i * 32] = 7; 
        }
    }
}

void launch_stress_memory(cudaStream_t stream, unsigned int* device_array, int bytesize, int stride_bytes, unsigned long long run_time) {
    int num_lines = bytesize / stride_bytes; 
    dim3 blocks(20);
    dim3 threads(256); // SHRINK: Fits perfectly onto the 8 masked SMs
    memoryKernel_Write<<<blocks, threads, 0, stream>>>(device_array, num_lines, run_time);
}