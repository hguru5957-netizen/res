#pragma once
#include <cuda_runtime.h>

struct YoloWorkspace;

// Loads cfg + weights once. Returns nullptr on failure.
YoloWorkspace* yoloWorkspaceCreate(const char* cfg_path,
                                   const char* weights_path);

// Release everything. Safe with nullptr.
void yoloWorkspaceDestroy(YoloWorkspace* ws);

// Runs one forward pass on `stream`. No allocations, no sync.
void launch_yolo(cudaStream_t stream, YoloWorkspace* ws);