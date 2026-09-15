#include "common.cuh"

__global__ void computeKernel_INT32(unsigned long long run_time) {
    int32_t e = 0; 
    int32_t e_op_add = 1; 
    
    unsigned long long clock_begin = clock64();
    
    while ((clock64() - clock_begin) < run_time) {
        #pragma unroll 32
        for (int access_count = 0; access_count < 10000; access_count++) { 
            // volatile prevents compiler from deleting the loop
            asm volatile("add.s32 %0, %0, %1;" : "+r"(e) : "r"(e_op_add));
        }
    }
}

void launch_stress_compute(cudaStream_t stream, unsigned long long run_time) {
    dim3 blocks(20); 
    dim3 threads(256); // SHRINK: Fits perfectly onto the 8 masked SMs
    computeKernel_INT32<<<blocks, threads, 0, stream>>>(run_time);
}