#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "libsmctrl.h"

// Your custom headers
#include "victims.cuh"
#include "enemies.cuh"

// --- ERROR CHECKING MACRO ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- TASK DISPATCHER ---
// Routes the integer ID to the correct wrapper, passing pre-allocated memory
void launch_task(int id, cudaStream_t stream, float* d_A, float* d_B, float* d_C) {
    switch (id) {
        case 1: launch_matrixMul(stream, d_A, d_B, d_C); break;
        case 2: launch_vectorAdd(stream, d_A, d_B, d_C, 50000); break;
        
        // FIX: Cast the float pointers to unsigned int pointers for the imaging task
        case 3: launch_stereoDisparity(stream, (unsigned int*)d_A, (unsigned int*)d_B, (unsigned int*)d_C); break; 
        
        // Synthetic Enemies (configured for ~2 seconds to ensure overlap)
        case 4: launch_stress_compute(stream, 2000000000ULL); break; 
        case 5: launch_stress_memory(stream, (unsigned int*)d_A, 16*1024*1024, 128, 2000000000ULL); break;
        
        default: std::cerr << "Invalid Task ID: " << id << "\n"; exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./profiler <TaskA_ID> <TaskB_ID>\n";
        return EXIT_FAILURE;
    }

    int taskA = atoi(argv[1]);
    int taskB = atoi(argv[2]);

    // 1. UPFRONT MEMORY ALLOCATION (64 MB per buffer to cover largest workloads)
    size_t pool_size = 64 * 1024 * 1024;
    float *d_bufA, *d_bufB, *d_bufC;
    float *d_bufD, *d_bufE, *d_bufF;
    
    CUDA_CHECK(cudaMalloc(&d_bufA, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufB, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufC, pool_size));
    
    // Separate buffers for Task B so they don't corrupt Task A's memory
    CUDA_CHECK(cudaMalloc(&d_bufD, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufE, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufF, pool_size));

    // 2. STREAM CREATION & HARDWARE PARTITIONING
    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    // IMPORTANT: Actual GPC masks for your Jetson Orin (8 TPCs interleaved)
    uint64_t mask_partition_0 = 0xaaULL; // GPC 0
    uint64_t mask_partition_1 = 0x55ULL; // GPC 1

    //libsmctrl_set_stream_mask(streamA, mask_partition_0);
    //libsmctrl_set_stream_mask(streamB, mask_partition_1);

    // 3. TIMER SETUP
    cudaEvent_t startA, stopA, startB, stopB;
    CUDA_CHECK(cudaEventCreate(&startA)); CUDA_CHECK(cudaEventCreate(&stopA));
    CUDA_CHECK(cudaEventCreate(&startB)); CUDA_CHECK(cudaEventCreate(&stopB));

    float time_A_solo = 0.0f, time_B_solo = 0.0f;
    float time_A_corun = 0.0f, time_B_corun = 0.0f;

    // 4. WARMUP (Force lazy initialization before timing)
    for (int i = 0; i < 3; i++) {
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC);
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // RUN 1: SOLO PROFILING (20 Iterations)
    // -------------------------------------------------------------
    int num_trials = 5;
    std::vector<float> solo_A(num_trials), solo_B(num_trials);
    std::vector<float> corun_A(num_trials), corun_B(num_trials);

    for (int i = 0; i < num_trials; i++) {
        CUDA_CHECK(cudaEventRecord(startA, streamA));
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC);
        CUDA_CHECK(cudaEventRecord(stopA, streamA));
        CUDA_CHECK(cudaEventSynchronize(stopA));
        CUDA_CHECK(cudaEventElapsedTime(&solo_A[i], startA, stopA));

        CUDA_CHECK(cudaEventRecord(startB, streamB));
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF);
        CUDA_CHECK(cudaEventRecord(stopB, streamB));
        CUDA_CHECK(cudaEventSynchronize(stopB));
        CUDA_CHECK(cudaEventElapsedTime(&solo_B[i], startB, stopB));
    }

    // -------------------------------------------------------------
    // RUN 2: CONCURRENT PROFILING (20 Iterations)
    // -------------------------------------------------------------
    for (int i = 0; i < num_trials; i++) {
        CUDA_CHECK(cudaEventRecord(startB, streamB));
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF);
        CUDA_CHECK(cudaEventRecord(stopB, streamB)); // Moved up!
        
        CUDA_CHECK(cudaEventRecord(startA, streamA));
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC);
        CUDA_CHECK(cudaEventRecord(stopA, streamA));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventElapsedTime(&corun_A[i], startA, stopA));
        CUDA_CHECK(cudaEventElapsedTime(&corun_B[i], startB, stopB));
    }

    // -------------------------------------------------------------
    // STATISTICAL EXTRACTION
    // -------------------------------------------------------------
    std::sort(solo_A.begin(), solo_A.end());
    std::sort(solo_B.begin(), solo_B.end());
    std::sort(corun_A.begin(), corun_A.end());
    std::sort(corun_B.begin(), corun_B.end());

    // Median (50th percentile) for Solo Baseline
    time_A_solo = solo_A[num_trials / 2];
    time_B_solo = solo_B[num_trials / 2];

    // 90th Percentile for Concurrent High-Stress
    int p90_index = (int)(num_trials * 0.90);
    time_A_corun = corun_A[p90_index];
    time_B_corun = corun_B[p90_index];
    
    // -------------------------------------------------------------
    // 5. OUTPUT RESULTS (CSV Friendly)
    // -------------------------------------------------------------
    float ratio_A = time_A_corun / time_A_solo;
    float ratio_B = time_B_corun / time_B_solo;

    // Format: TaskA, TaskB, SoloA, CoRunA, RatioA, SoloB, CoRunB, RatioB
    std::cout << std::fixed << std::setprecision(4) 
              << taskA << "," << taskB << "," 
              << time_A_solo << "," << time_A_corun << "," << ratio_A << ","
              << time_B_solo << "," << time_B_corun << "," << ratio_B << "\n";

    // 6. CLEANUP
    CUDA_CHECK(cudaFree(d_bufA)); CUDA_CHECK(cudaFree(d_bufB)); CUDA_CHECK(cudaFree(d_bufC));
    CUDA_CHECK(cudaFree(d_bufD)); CUDA_CHECK(cudaFree(d_bufE)); CUDA_CHECK(cudaFree(d_bufF));
    cudaStreamDestroy(streamA); cudaStreamDestroy(streamB);
    
    return 0;
}