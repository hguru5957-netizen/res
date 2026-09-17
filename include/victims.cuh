#pragma once
#include <cuda_runtime.h>

void launch_vectorAdd(cudaStream_t stream, float* d_A, float* d_B, float* d_C, int numElements);
void launch_matrixMul(cudaStream_t stream, float* d_A, float* d_B, float* d_C);
void launch_stereoDisparity(cudaStream_t stream, unsigned int* d_img0, unsigned int* d_img1, unsigned int* d_odata);
void launch_hsOpticalFlow(cudaStream_t stream, float* d_A, float* d_B, float* d_C);