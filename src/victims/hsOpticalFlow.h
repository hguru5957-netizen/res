#pragma once
#include <cuda_runtime.h>

// Full definition lives in HSOpticalFlow/flowCUDA.h — forward-declare here
// so callers (main.cu) only need this header, not the Darknet-style includes.
struct HSOFWorkspace;

// Allocate every buffer the optical-flow algorithm needs.
// Called once, from main(), before the trial loops.
// Returns nullptr on failure.
HSOFWorkspace* hsOFWorkspaceCreate();

// Release everything allocated by hsOFWorkspaceCreate().
// Safe to call with nullptr.
void hsOFWorkspaceDestroy(HSOFWorkspace* ws);

// Run one optical-flow computation on `stream`.
//   d_A -> I0 (device, stride*height floats, stride == width in the workspace)
//   d_B -> I1 (device, same size as I0)
//   d_C -> output u (device); output v is placed at d_C + stride*height
//
// No allocations, no frees, no device-wide syncs. Timed by the caller's
// cudaEventRecord(stop, stream) after this returns.
void launch_hsOpticalFlow(cudaStream_t stream,
                          const float* d_A, const float* d_B, float* d_C,
                          HSOFWorkspace* ws);