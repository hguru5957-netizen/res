#include "common.cuh"
#include <stdint.h>


// ---------------------------------------------------------
// COMPUTE ENEMY KERNEL
// ---------------------------------------------------------
__global__ void computeKernel_INT32(unsigned long long run_time) {
    int32_t e = 0; 
    int32_t e_op_add = 1; 
    
    unsigned long long clock_begin = gclock64();
    
    // Loop until the global timer exceeds the requested nanoseconds
    while ((gclock64() - clock_begin) < run_time) {
        #pragma unroll 32
        for (int access_count = 0; access_count < 10000; access_count++) { 
            // volatile prevents compiler from optimizing this away
            asm volatile("add.s32 %0, %0, %1;" : "+r"(e) : "r"(e_op_add));
        }
    }
}

// ---------------------------------------------------------
// COMPUTE ENEMY LAUNCHER
// ---------------------------------------------------------
// allocated_sms defaults to 8 (1 GPC on Orin), but can be adjusted 
// if your partition size changes.
void launch_stress_compute(cudaStream_t stream, unsigned long long run_time, int allocated_sms) {
    // Launch 2 full blocks per allocated SM to guarantee even WDU distribution
    dim3 blocks(allocated_sms * 2); 
    // MAXIMIZE: 1024 threads (32 warps) per block to saturate the SM compute units
    dim3 threads(1024); 
    
    computeKernel_INT32<<<blocks, threads, 0, stream>>>(run_time);
}