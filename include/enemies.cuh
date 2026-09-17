#pragma once
#include <cuda_runtime.h>

// Add the allocated_sms parameter here with the default value of 8
void launch_stress_compute(cudaStream_t stream, unsigned long long run_time, int allocated_sms = 8);

void launch_stress_memory(cudaStream_t stream, unsigned int* device_array, int bytesize, int stride_bytes, unsigned long long run_time, int allocated_sms = 8);