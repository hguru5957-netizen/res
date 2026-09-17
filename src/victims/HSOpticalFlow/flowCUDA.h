/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 * (license header unchanged)
 */

#ifndef FLOW_CUDA_H
#define FLOW_CUDA_H

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Pre-allocated workspace for HSOpticalFlow.
// All device buffers are allocated once in hsOFWorkspaceCreate() and reused
// every launch, so ComputeFlowCUDA performs no cudaMalloc / cudaFree /
// cudaDeviceSynchronize and stays safe inside a timing loop.
// ---------------------------------------------------------------------------
struct HSOFWorkspace {
    // Geometry
    int width;
    int height;
    int stride;
    int nLevels;

    // Pyramid buffers (device pointers)
    const float** pI0;
    const float** pI1;

    // Per-level size bookkeeping (host side)
    int* pW;
    int* pH;
    int* pS;

    // Base-size scratch (device) — base = stride * height * sizeof(float)
    float* d_tmp;

    float* d_du0;
    float* d_dv0;
    float* d_du1;
    float* d_dv1;

    float* d_Ix;
    float* d_Iy;
    float* d_Iz;

    float* d_u;
    float* d_v;
    float* d_nu;
    float* d_nv;
};

// ---------------------------------------------------------------------------
// Compute dense optical flow between I0 and I1 on the caller's stream.
// I0, I1, u, v must be DEVICE pointers. No allocation, no free, no sync.
// ---------------------------------------------------------------------------
void ComputeFlowCUDA(cudaStream_t stream,
                     const float *I0,
                     const float *I1,
                     float alpha,
                     int nWarpIters,
                     int nSolverIters,
                     float *u,
                     float *v,
                     HSOFWorkspace* ws);

#endif // FLOW_CUDA_H