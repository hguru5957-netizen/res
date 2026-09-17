#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "libsmctrl.h"
#include <cstdlib>
#include <cstdint>
#include "victims.cuh"
#include "enemies.cuh"
#include "hsOpticalFlow.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" \
                      << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void launch_task(int id, cudaStream_t stream,
                 float* d_A, float* d_B, float* d_C,
                 int allocated_sms,
                 HSOFWorkspace* hsWs)
{
    switch (id) {
        case 1:
            launch_matrixMul(stream, d_A, d_B, d_C);
            break;
        case 2:
            launch_vectorAdd(stream, d_A, d_B, d_C, 50000);
            break;
        case 3:
            launch_stereoDisparity(
                stream,
                reinterpret_cast<unsigned int*>(d_A),
                reinterpret_cast<unsigned int*>(d_B),
                reinterpret_cast<unsigned int*>(d_C)
            );
            break;
        case 4:
            launch_stress_compute(stream, 2000000000ULL, allocated_sms);
            break;
        case 5:
            launch_stress_memory(
                stream,
                reinterpret_cast<unsigned int*>(d_A),
                16 * 1024 * 1024,
                128,
                2000000000ULL,
                allocated_sms
            );
            break;
        case 6:
            launch_hsOpticalFlow(stream, d_A, d_B, d_C, hsWs);
            break;
        default:
            std::cerr << "Invalid Task ID: " << id << std::endl;
            exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: ./profiler <TaskA_ID> <TaskB_ID>" << std::endl;
        return EXIT_FAILURE;
    }

    int taskA = std::atoi(argv[1]);
    int taskB = std::atoi(argv[2]);

    size_t pool_size = 64 * 1024 * 1024;

    float* d_bufA;
    float* d_bufB;
    float* d_bufC;
    float* d_bufD;
    float* d_bufE;
    float* d_bufF;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bufA), pool_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bufB), pool_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bufC), pool_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bufD), pool_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bufE), pool_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bufF), pool_size));

    cudaStream_t streamA;
    cudaStream_t streamB;

    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    HSOFWorkspace* hsWsA = hsOFWorkspaceCreate();
    HSOFWorkspace* hsWsB = hsOFWorkspaceCreate();
    if (!hsWsA || !hsWsB) {
        std::cerr << "Failed to create HSOpticalFlow workspace" << std::endl;
        return EXIT_FAILURE;
    }

    uint32_t num_gpcs = 0;
    uint64_t* tpcs_for_gpc = nullptr;

    libsmctrl_get_gpc_info(&num_gpcs, &tpcs_for_gpc, 0);

    if (num_gpcs < 2) {
        std::cerr << "Error: Not enough GPCs to partition!" << std::endl;

        hsOFWorkspaceDestroy(hsWsA);
        hsOFWorkspaceDestroy(hsWsB);

        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);

        cudaFree(d_bufA);
        cudaFree(d_bufB);
        cudaFree(d_bufC);
        cudaFree(d_bufD);
        cudaFree(d_bufE);
        cudaFree(d_bufF);

        return EXIT_FAILURE;
    }

    int tpcs_in_gpc0 = __builtin_popcountll(tpcs_for_gpc[0]);
    int tpcs_in_gpc1 = __builtin_popcountll(tpcs_for_gpc[1]);

    int sms_in_gpc0 = tpcs_in_gpc0 * 2;
    int sms_in_gpc1 = tpcs_in_gpc1 * 2;

    std::cout
        << "GPC count: " << num_gpcs << "\n"
        << "GPC 0 TPCs: " << tpcs_in_gpc0 << "\n"
        << "GPC 1 TPCs: " << tpcs_in_gpc1 << "\n"
        << "GPC 0 SMs:  " << sms_in_gpc0 << "\n"
        << "GPC 1 SMs:  " << sms_in_gpc1 << "\n";

    uint64_t disable_mask_A = ~tpcs_for_gpc[0];
    uint64_t disable_mask_B = ~tpcs_for_gpc[1];

    libsmctrl_set_stream_mask(streamA, disable_mask_A);
    libsmctrl_set_stream_mask(streamB, disable_mask_B);

    cudaEvent_t startA;
    cudaEvent_t stopA;
    cudaEvent_t startB;
    cudaEvent_t stopB;

    CUDA_CHECK(cudaEventCreate(&startA));
    CUDA_CHECK(cudaEventCreate(&stopA));
    CUDA_CHECK(cudaEventCreate(&startB));
    CUDA_CHECK(cudaEventCreate(&stopB));

    const int num_trials = 10;

    std::vector<float> solo_A(num_trials);
    std::vector<float> solo_B(num_trials);
    std::vector<float> corun_A(num_trials);
    std::vector<float> corun_B(num_trials);

    for (int i = 0; i < 3; i++) {
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC, sms_in_gpc0, hsWsA);
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF, sms_in_gpc1, hsWsB);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_trials; i++) {
        CUDA_CHECK(cudaEventRecord(startA, streamA));
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC, sms_in_gpc0, hsWsA);
        CUDA_CHECK(cudaEventRecord(stopA, streamA));
        CUDA_CHECK(cudaEventSynchronize(stopA));
        CUDA_CHECK(cudaEventElapsedTime(&solo_A[i], startA, stopA));

        CUDA_CHECK(cudaEventRecord(startB, streamB));
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF, sms_in_gpc1, hsWsB);
        CUDA_CHECK(cudaEventRecord(stopB, streamB));
        CUDA_CHECK(cudaEventSynchronize(stopB));
        CUDA_CHECK(cudaEventElapsedTime(&solo_B[i], startB, stopB));
    }

    for (int i = 0; i < num_trials; i++) {
        CUDA_CHECK(cudaEventRecord(startB, streamB));
        launch_task(taskB, streamB, d_bufD, d_bufE, d_bufF, sms_in_gpc1, hsWsB);
        CUDA_CHECK(cudaEventRecord(stopB, streamB));

        CUDA_CHECK(cudaEventRecord(startA, streamA));
        launch_task(taskA, streamA, d_bufA, d_bufB, d_bufC, sms_in_gpc0, hsWsA);
        CUDA_CHECK(cudaEventRecord(stopA, streamA));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventElapsedTime(&corun_A[i], startA, stopA));
        CUDA_CHECK(cudaEventElapsedTime(&corun_B[i], startB, stopB));
    }

    std::sort(solo_A.begin(), solo_A.end());
    std::sort(solo_B.begin(), solo_B.end());
    std::sort(corun_A.begin(), corun_A.end());
    std::sort(corun_B.begin(), corun_B.end());

    int median_index = num_trials / 2;

    float time_A_solo = solo_A[median_index];
    float time_B_solo = solo_B[median_index];

    int p90_index = static_cast<int>(std::ceil(0.90 * num_trials)) - 1;

    if (p90_index < 0)
        p90_index = 0;

    if (p90_index >= num_trials)
        p90_index = num_trials - 1;

    float time_A_corun = corun_A[p90_index];
    float time_B_corun = corun_B[p90_index];

    float ratio_A = time_A_corun / time_A_solo;
    float ratio_B = time_B_corun / time_B_solo;

    std::cout
        << std::fixed
        << std::setprecision(4)
        << taskA << ","
        << taskB << ","
        << time_A_solo << ","
        << time_A_corun << ","
        << ratio_A << ","
        << time_B_solo << ","
        << time_B_corun << ","
        << ratio_B << "\n";

    hsOFWorkspaceDestroy(hsWsA);
    hsOFWorkspaceDestroy(hsWsB);

    CUDA_CHECK(cudaEventDestroy(startA));
    CUDA_CHECK(cudaEventDestroy(stopA));
    CUDA_CHECK(cudaEventDestroy(startB));
    CUDA_CHECK(cudaEventDestroy(stopB));

    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));

    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));
    CUDA_CHECK(cudaFree(d_bufC));
    CUDA_CHECK(cudaFree(d_bufD));
    CUDA_CHECK(cudaFree(d_bufE));
    CUDA_CHECK(cudaFree(d_bufF));

    return EXIT_SUCCESS;
}