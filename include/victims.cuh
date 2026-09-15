#pragma once
#include <cuda_runtime.h>

void launch_vectorAdd(cudaStream_t stream);
void launch_matrixMul(cudaStream_t stream);
void launch_stereoDisparity(cudaStream_t stream);