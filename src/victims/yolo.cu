#include "yolo.h"
#include <cstdio>
#include <cstring>
#include <unistd.h>     // dup, dup2

#include "darknet.h"
#include "parser.h"
#include "network.h"

extern "C" void darknet_set_stream(cudaStream_t s);

struct YoloWorkspace {
    network net;
    float*  input_host;
    bool    valid;
};

YoloWorkspace* yoloWorkspaceCreate(const char* cfg_path,
                                   const char* weights_path)
{
    YoloWorkspace* ws = new YoloWorkspace();
    ws->valid      = false;
    ws->input_host = nullptr;

    // ---- Silence Darknet's verbose banner during setup -------------------
    fflush(stdout);
    int saved_stdout = dup(STDOUT_FILENO);
    FILE* devnull = freopen("/dev/null", "w", stdout);

    ws->net = parse_network_cfg_custom((char*)cfg_path, 1, 1);

    if (ws->net.n > 0 && ws->net.layers != nullptr) {
        load_weights(&ws->net, (char*)weights_path);
        ws->net.batch = 1;
    }

    // Restore stdout
    fflush(stdout);
    dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);
    if (devnull) clearerr(stdout);

    if (ws->net.n <= 0 || ws->net.layers == nullptr) {
        delete ws;
        return nullptr;
    }

    // Pinned host input buffer, zero-filled
    const size_t input_bytes =
        (size_t)ws->net.w * ws->net.h * ws->net.c * sizeof(float);

    cudaError_t err = cudaHostAlloc((void**)&ws->input_host, input_bytes,
                                    cudaHostAllocDefault);
    if (err != cudaSuccess) {
        delete ws;
        return nullptr;
    }
    memset(ws->input_host, 0, input_bytes);

    cudaDeviceSynchronize();
    ws->valid = true;
    return ws;
}

void yoloWorkspaceDestroy(YoloWorkspace* ws)
{
    if (!ws) return;

    // Deliberately skip free_network() and cudaFreeHost().
    //
    // Darknet's parse_network_cfg_custom stores the CUDA stream and cuBLAS
    // handle in __thread globals that are shared between both workspaces
    // (they were created on the same thread). Calling free_network on either
    // one then leaves the other holding dangling stream state, and the
    // second cleanup crashes. Since the profiler process is short-lived,
    // we let OS + CUDA-context teardown reclaim everything instead.
    //
    // If you ever embed this in a long-running process, revisit: you'll
    // need per-workspace stream/cublas isolation, e.g. one worker thread
    // per workspace, so the __thread state is distinct.

    delete ws;
}

void launch_yolo(cudaStream_t stream, YoloWorkspace* ws)
{
    darknet_set_stream(stream);
    network_predict_gpu(ws->net, ws->input_host);
}