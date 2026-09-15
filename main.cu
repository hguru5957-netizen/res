#include <iostream>
#include <cuda_runtime.h>
#include "libsmctrl.h"
#include "victims.cuh"
#include "enemies.cuh"

// Dispatcher to route IDs to the correct function
void launch_task(int id, cudaStream_t stream) {
    if (id == 1) launch_matrixMul(stream);
    else if (id == 2) launch_vectorAdd(stream);
    // Add other tasks here...
}

int main(int argc, char** argv) {
    int taskA = atoi(argv[1]);
    int taskB = atoi(argv[2]);

    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // Apply hardware partitioning masks (replace with your Orin's GPC masks)
    libsmctrl_set_stream_mask(streamA, 0x000000FFULL);
    libsmctrl_set_stream_mask(streamB, 0x0000FF00ULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0.0f;

    // Launch concurrent profiling
    cudaEventRecord(start, streamA);
    launch_task(taskA, streamA); // Victim
    launch_task(taskB, streamB); // Enemy competitor
    cudaEventRecord(stop, streamA);
    
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Task " << taskA << " vs Task " << taskB << " Time: " << ms << " ms\n";
    return 0;
}