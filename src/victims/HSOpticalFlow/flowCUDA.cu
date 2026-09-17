#include "common.h"

#include "downscaleKernel.cuh"
#include "upscaleKernel.cuh"
#include "warpingKernel.cuh"
#include "derivativesKernel.cuh"
#include "solverKernel.cuh"
#include "addKernel.cuh"

#include "flowCUDA.h"

void ComputeFlowCUDA(cudaStream_t stream,
                     const float *I0, const float *I1,
                     float alpha,
                     int nWarpIters, int nSolverIters,
                     float *u, float *v,
                     HSOFWorkspace* ws)
{
    const int width   = ws->width;
    const int height  = ws->height;
    const int stride  = ws->stride;
    const int nLevels = ws->nLevels;

    const float **pI0 = ws->pI0;
    const float **pI1 = ws->pI1;
    int *pW = ws->pW;
    int *pH = ws->pH;
    int *pS = ws->pS;

    float *d_tmp = ws->d_tmp;
    float *d_du0 = ws->d_du0, *d_dv0 = ws->d_dv0;
    float *d_du1 = ws->d_du1, *d_dv1 = ws->d_dv1;
    float *d_Ix  = ws->d_Ix,  *d_Iy  = ws->d_Iy,  *d_Iz = ws->d_Iz;
    float *d_u   = ws->d_u,   *d_v   = ws->d_v;
    float *d_nu  = ws->d_nu,  *d_nv  = ws->d_nv;

    const int dataSize = stride * height * sizeof(float);

    // ---- Upload I0 / I1 into the top pyramid level (D2D on caller's stream)
    int currentLevel = nLevels - 1;
    checkCudaErrors(cudaMemcpyAsync((void *)pI0[currentLevel], I0, dataSize,
                                    cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync((void *)pI1[currentLevel], I1, dataSize,
                                    cudaMemcpyDeviceToDevice, stream));

    // ---- Build pyramid (down to level 0) ---------------------------------
    for (currentLevel = nLevels - 1; currentLevel > 0; --currentLevel)
    {
        Downscale(stream,
                  pI0[currentLevel], pW[currentLevel], pH[currentLevel],
                  pS[currentLevel],
                  pW[currentLevel - 1], pH[currentLevel - 1], pS[currentLevel - 1],
                  (float *)pI0[currentLevel - 1]);

        Downscale(stream,
                  pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                  pS[currentLevel],
                  pW[currentLevel - 1], pH[currentLevel - 1], pS[currentLevel - 1],
                  (float *)pI1[currentLevel - 1]);
    }

    // ---- Reset flow at the coarsest level --------------------------------
    checkCudaErrors(cudaMemsetAsync(d_u, 0, dataSize, stream));
    checkCudaErrors(cudaMemsetAsync(d_v, 0, dataSize, stream));

    // ---- Iterate up the pyramid ------------------------------------------
    for (currentLevel = 0; currentLevel < nLevels; ++currentLevel)
    {
        for (int warpIter = 0; warpIter < nWarpIters; ++warpIter)
        {
            checkCudaErrors(cudaMemsetAsync(d_du0, 0, dataSize, stream));
            checkCudaErrors(cudaMemsetAsync(d_dv0, 0, dataSize, stream));
            checkCudaErrors(cudaMemsetAsync(d_du1, 0, dataSize, stream));
            checkCudaErrors(cudaMemsetAsync(d_dv1, 0, dataSize, stream));

            WarpImage(stream,
                      pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                      pS[currentLevel], d_u, d_v, d_tmp);

            ComputeDerivatives(stream,
                               pI0[currentLevel], d_tmp,
                               pW[currentLevel], pH[currentLevel], pS[currentLevel],
                               d_Ix, d_Iy, d_Iz);

            for (int iter = 0; iter < nSolverIters; ++iter)
            {
                SolveForUpdate(stream,
                               d_du0, d_dv0, d_Ix, d_Iy, d_Iz,
                               pW[currentLevel], pH[currentLevel], pS[currentLevel],
                               alpha, d_du1, d_dv1);

                Swap(d_du0, d_du1);
                Swap(d_dv0, d_dv1);
            }

            Add(stream, d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
            Add(stream, d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
        }

        if (currentLevel != nLevels - 1)
        {
            float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];
            float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

            Upscale(stream,
                    d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
                    scaleX, d_nu);

            Upscale(stream,
                    d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
                    pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
                    scaleY, d_nv);

            Swap(d_u, d_nu);
            Swap(d_v, d_nv);
        }
    }

    // ---- Emit final flow (D2D on caller's stream). No sync, no frees. ----
    checkCudaErrors(cudaMemcpyAsync(u, d_u, dataSize,
                                    cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(v, d_v, dataSize,
                                    cudaMemcpyDeviceToDevice, stream));
}