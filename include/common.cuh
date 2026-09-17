#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
static __device__ __inline__ unsigned long long int gclock64() {
    unsigned long long int rv;
    asm volatile ( "mov.u64 %0, %%globaltimer;" : "=l"(rv) );
    return rv;
}