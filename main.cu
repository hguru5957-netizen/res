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
void launch_task(int id, cudaStream_t stream, float* d_A, float* d_B, float* d_C) {
    switch (id) {
        case 1: launch_matrixMul(stream, d_A, d_B, d_C); break;
        case 2: launch_vectorAdd(stream, d_A, d_B, d_C, 50000); break;
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

    // 1. UPFRONT MEMORY ALLOCATION
    size_t pool_size = 64 * 1024 * 1024;
    float *d_bufA, *d_bufB, *d_bufC;
    float *d_bufD, *d_bufE, *d_bufF;
    
    CUDA_CHECK(cudaMalloc(&d_bufA, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufB, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufC, pool_size));
    
    CUDA_CHECK(cudaMalloc(&d_bufD, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufE, pool_size));
    CUDA_CHECK(cudaMalloc(&d_bufF, pool_size));

    // 2. STREAM CREATION & HARDWARE PARTITIONING
    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    // DYNAMIC GPC MASK RETRIEVAL
    uint32_t num_gpcs;
    uint64_t *tpcs_for_gpc;
    // 0 is the device ID for the Jetson
    libsmctrl_get_gpc_info(&num_gpcs, &tpcs_for_gpc, 0); 

    if (num_gpcs < 2) {
        std::cerr << "Error: Not enough GPCs to partition!" << std::endl;
        return EXIT_FAILURE;
    }

    // To allow a stream to use GPC 0, we must DISABLE everything else.
    // We do this by bitwise inverting (~) the TPCs associated with GPC 0.
    uint64_t disable_mask_A = ~tpcs_for_gpc[0]; 
    uint64_t disable_mask_B = ~tpcs_for_gpc[1]; 

    libsmctrl_set_stream_mask(streamA, disable_mask_A);
    libsmctrl_set_stream_mask(streamB, disable_mask_B);

    // 3. TIMER SETUP
    cudaEvent_t startA, stopA, startB, stopB;
    CUDA_CHECK(cudaEventCreate(&startA)); CUDA_CHECK(cudaEventCreate(&stopA));
    CUDA_CHECK(cudaEventCreate(&startB)); CUDA_CHECK(cudaEventCreate(&stopB));

    float time_A_solo = 0.0f, time_B_solo = 0.0f;
    float time_A_corun = 0.0f, time_B_corun = 0.0f;

    // 4. WARMUP
    for (int i = 0; i < 3; i++) {
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC);
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // RUN 1: SOLO PROFILING (Increased to 10 Iterations)
    // -------------------------------------------------------------
    int num_trials = 10;
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
    // RUN 2: CONCURRENT PROFILING 
    // -------------------------------------------------------------
    for (int i = 0; i < num_trials; i++) {
        CUDA_CHECK(cudaEventRecord(startB, streamB));
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF);
        CUDA_CHECK(cudaEventRecord(stopB, streamB)); 
        
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
    // 5. OUTPUT RESULTS
    // -------------------------------------------------------------
    float ratio_A = time_A_corun / time_A_solo;
    float ratio_B = time_B_corun / time_B_solo;

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