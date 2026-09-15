#include <stdio.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * BLOCK_SIZE * by;
  int aEnd = aBegin + wA - 1;
  int aStep = BLOCK_SIZE;

  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * wB;

  float Csub = 0;

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
      Csub += As[ty][k] * Bs[k][tx];

    __syncthreads();
  }

  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void launch_matrixMul(cudaStream_t stream, float* d_A, float* d_B, float* d_C) {
  int block_size = 32;

  dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
  dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  for (int i = 0; i < 1000; ++i) {
    MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(
        d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
}
