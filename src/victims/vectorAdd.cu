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
#include <helper_cuda.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host launch routine
 */
void launch_vectorAdd(cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;

    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Asynchronous host-to-device copies using the supplied stream
    printf("Copy input data from the host memory to the CUDA device\n");

    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n",
           blocksPerGrid, threadsPerBlock);

    // Launch kernel on the supplied stream
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_A, d_B, d_C, numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Asynchronous device-to-host copy using the supplied stream
    printf("Copy output data from the CUDA device to the host memory\n");

    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for the stream operations to complete before accessing h_C
    err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize CUDA stream (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
}
