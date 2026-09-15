/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 */

#include <cuda_runtime.h>
#include "stereoDisparity_kernel.cuh"

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void launch_stereoDisparity(cudaStream_t stream,
                            unsigned int* d_img0,
                            unsigned int* d_img1,
                            unsigned int* d_odata)
{
    unsigned int w = 640;
    unsigned int h = 533;

    int minDisp = -16;
    int maxDisp = 0;

    dim3 numThreads(blockSize_x, blockSize_y, 1);
    dim3 numBlocks(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));

    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

    cudaTextureObject_t tex2Dleft, tex2Dright;
    cudaResourceDesc texRes = {};
    cudaTextureDesc texDescr = {};

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = d_img0;
    texRes.res.pitch2D.desc = ca_desc0;
    texRes.res.pitch2D.width = w;
    texRes.res.pitch2D.height = h;
    texRes.res.pitch2D.pitchInBytes = w * 4;

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex2Dleft, &texRes, &texDescr, NULL);

    texRes = {};
    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = d_img1;
    texRes.res.pitch2D.desc = ca_desc1;
    texRes.res.pitch2D.width = w;
    texRes.res.pitch2D.height = h;
    texRes.res.pitch2D.pitchInBytes = w * 4;

    cudaCreateTextureObject(&tex2Dright, &texRes, &texDescr, NULL);
    // Boost execution time
    for (int i = 0; i < 50; i++) {
        stereoDisparityKernel<<<numBlocks, numThreads, 0, stream>>>(
            d_img0, d_img1, d_odata, w, h,
            minDisp, maxDisp, tex2Dleft, tex2Dright);
    }
    cudaDestroyTextureObject(tex2Dleft);
    cudaDestroyTextureObject(tex2Dright);
}