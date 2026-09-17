#include "hsOpticalFlow.h"
#include "HSOpticalFlow/flowCUDA.h"
#include "HSOpticalFlow/common.h"    // for iAlignUp

#define HSOF_ALPHA     0.001f
#define HSOF_N_WARP    3
#define HSOF_N_SOLVER  5
#define HSOF_N_LEVELS  5
#define HSOF_W         640
#define HSOF_H         480

HSOFWorkspace* hsOFWorkspaceCreate()
{
    HSOFWorkspace* ws = new HSOFWorkspace();
    ws->width   = HSOF_W;
    ws->height  = HSOF_H;
    ws->stride  = HSOF_W;
    ws->nLevels = HSOF_N_LEVELS;

    const int dataSize = ws->stride * ws->height * sizeof(float);

    // Base-size scratch
    cudaMalloc(&ws->d_tmp, dataSize);
    cudaMalloc(&ws->d_du0, dataSize);
    cudaMalloc(&ws->d_dv0, dataSize);
    cudaMalloc(&ws->d_du1, dataSize);
    cudaMalloc(&ws->d_dv1, dataSize);
    cudaMalloc(&ws->d_Ix,  dataSize);
    cudaMalloc(&ws->d_Iy,  dataSize);
    cudaMalloc(&ws->d_Iz,  dataSize);
    cudaMalloc(&ws->d_u,   dataSize);
    cudaMalloc(&ws->d_v,   dataSize);
    cudaMalloc(&ws->d_nu,  dataSize);
    cudaMalloc(&ws->d_nv,  dataSize);

    // Pyramid buffers (sized once, reused every call)
    ws->pI0 = new const float*[ws->nLevels];
    ws->pI1 = new const float*[ws->nLevels];
    ws->pW  = new int[ws->nLevels];
    ws->pH  = new int[ws->nLevels];
    ws->pS  = new int[ws->nLevels];

    int cur = ws->nLevels - 1;
    cudaMalloc(ws->pI0 + cur, dataSize);
    cudaMalloc(ws->pI1 + cur, dataSize);
    ws->pW[cur] = ws->width;
    ws->pH[cur] = ws->height;
    ws->pS[cur] = ws->stride;

    for (; cur > 0; --cur) {
        int nw = ws->pW[cur] / 2;
        int nh = ws->pH[cur] / 2;
        int ns = iAlignUp(nw);
        cudaMalloc(ws->pI0 + cur - 1, ns * nh * sizeof(float));
        cudaMalloc(ws->pI1 + cur - 1, ns * nh * sizeof(float));
        ws->pW[cur - 1] = nw;
        ws->pH[cur - 1] = nh;
        ws->pS[cur - 1] = ns;
    }

    cudaDeviceSynchronize();   // one-time, off the hot path
    return ws;
}

void hsOFWorkspaceDestroy(HSOFWorkspace* ws)
{
    if (!ws) return;

    for (int i = 0; i < ws->nLevels; ++i) {
        cudaFree((void*)ws->pI0[i]);
        cudaFree((void*)ws->pI1[i]);
    }
    delete[] ws->pI0;
    delete[] ws->pI1;
    delete[] ws->pW;
    delete[] ws->pH;
    delete[] ws->pS;

    cudaFree(ws->d_tmp);
    cudaFree(ws->d_du0); cudaFree(ws->d_dv0);
    cudaFree(ws->d_du1); cudaFree(ws->d_dv1);
    cudaFree(ws->d_Ix);  cudaFree(ws->d_Iy);  cudaFree(ws->d_Iz);
    cudaFree(ws->d_u);   cudaFree(ws->d_v);
    cudaFree(ws->d_nu);  cudaFree(ws->d_nv);
    delete ws;
}

void launch_hsOpticalFlow(cudaStream_t stream,
                          const float* d_A, const float* d_B, float* d_C,
                          HSOFWorkspace* ws)
{
    float* d_u = d_C;
    float* d_v = d_C + (size_t)ws->stride * ws->height;

    ComputeFlowCUDA(stream,
                    d_A, d_B,
                    HSOF_ALPHA, HSOF_N_WARP, HSOF_N_SOLVER,
                    d_u, d_v,
                    ws);
}