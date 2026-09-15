#pragma once
#include <cuda_runtime.h>

void launch_stress_compute(cudaStream_t stream, unsigned long long run_time);
void launch_stress_memory(cudaStream_t stream, unsigned int* device_array, int bytesize, int stride_bytes, unsigned long long run_time);