/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        float a_val = A[i];
        float b_val = B[i];
        // The loop is now inside the GPU, preventing CPU queue overflow
        for (int iter = 0; iter < 5000; ++iter) {
            a_val = a_val + (b_val * 0.0001f); 
        }
        C[i] = a_val;
    }
}

void launch_vectorAdd(cudaStream_t stream, float* d_A, float* d_B, float* d_C, int numElements)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch exactly once!
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements);
}