// Smoke test for the adaptive {2,4,8} conditional-switch CUDA graph built by cuda_abi.cu.
// Exercises the CUDA 12 / CUDA 13 graph-capture shim on the target GPU without model weights.
//
// Build (from the repo root, CUDA 12.8+):
//   nvcc -O2 -std=c++20 -gencode arch=compute_120a,code=sm_120a --expt-relaxed-constexpr -I cuda \
//     tests/switch_graph_smoke.cu cuda/cuda_abi.cu cuda/kernels_gemv.cu cuda/kernels_inkling.cu \
//     cuda/kernels_dspark.cu cuda/kernels_prefill.cu -lcublas -ldl -o switch_graph_smoke
// Prints one line per case and exits 0 when every branch picks the expected width.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" {
int32_t squid_cuda_mem_get_info(size_t* free_bytes, size_t* total_bytes);
int32_t squid_adaptive_graph_create(
    const int32_t* d_anchor, const int32_t* d_drafts, const float* d_draft_conf, float threshold,
    int32_t* d_tokens, int32_t* d_chosen_width,
    const int32_t* d_accepted, int32_t* h_accepted,
    const int32_t* d_num_acc, int32_t* h_num_acc, int32_t* h_chosen_width,
    void* stream_raw, void** out_graph);
int32_t squid_adaptive_graph_branch_begin(void* graph_raw, int32_t branch_idx, void* stream_raw);
int32_t squid_adaptive_graph_branch_end(void* stream_raw);
int32_t squid_adaptive_graph_instantiate(void* graph_raw);
int32_t squid_adaptive_graph_launch(void* graph_raw, void* stream_raw);
int32_t squid_adaptive_graph_destroy(void* graph_raw);
}

#define CHECK(call)                                                                         \
    do {                                                                                    \
        int rc__ = (int)(call);                                                             \
        if (rc__ != 0) {                                                                    \
            fprintf(stderr, "FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #call, rc__, \
                    cudaGetErrorString((cudaError_t)rc__));                                 \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

int main() {
    size_t free_b = 0, total_b = 0;
    CHECK(squid_cuda_mem_get_info(&free_b, &total_b));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    size_t budget_free = 0, budget_total = 0;
    CHECK(cudaMemGetInfo(&budget_free, &budget_total));
    printf("device=%s sm_%d%d sms=%d runtime=%d driver=%d free=%.2f/%.2f GiB (cudaMemGetInfo %.2f)\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, rt, drv, free_b / 1073741824.0, total_b / 1073741824.0,
           budget_free / 1073741824.0);

    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int32_t *d_anchor, *d_drafts, *d_tokens, *d_width, *d_accepted, *d_num_acc;
    float* d_conf;
    CHECK(cudaMalloc(&d_anchor, 4));
    CHECK(cudaMalloc(&d_drafts, 7 * 4));
    CHECK(cudaMalloc(&d_conf, 7 * 4));
    CHECK(cudaMalloc(&d_tokens, 8 * 4));
    CHECK(cudaMalloc(&d_width, 4));
    CHECK(cudaMalloc(&d_accepted, 8 * 4));
    CHECK(cudaMalloc(&d_num_acc, 4));
    int32_t *h_accepted, *h_num_acc, *h_width;
    CHECK(cudaMallocHost(&h_accepted, 8 * 4));
    CHECK(cudaMallocHost(&h_num_acc, 4));
    CHECK(cudaMallocHost(&h_width, 4));

    const float threshold = 0.8f;
    void* graph = nullptr;
    CHECK(squid_adaptive_graph_create(d_anchor, d_drafts, d_conf, threshold, d_tokens, d_width, d_accepted,
                                      h_accepted, d_num_acc, h_num_acc, h_width, s, &graph));
    // Branch b writes byte (b + 1) into num_acc so the host can tell which body ran.
    for (int b = 0; b < 3; ++b) {
        CHECK(squid_adaptive_graph_branch_begin(graph, b, s));
        CHECK(cudaMemsetAsync(d_num_acc, b + 1, 4, s));
        CHECK(squid_adaptive_graph_branch_end(s));
    }
    CHECK(squid_adaptive_graph_instantiate(graph));

    const int anchor = 42;
    const int drafts[7] = {1, 2, 3, 4, 5, 6, 7};
    CHECK(cudaMemcpy(d_anchor, &anchor, 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_drafts, drafts, sizeof(drafts), cudaMemcpyHostToDevice));

    // confident-draft count -> (branch, width): 1 -> (0, 2), 3 -> (1, 4), 7 -> (2, 8)
    const int n_conf[3] = {1, 3, 7};
    const int want_width[3] = {2, 4, 8};
    int failures = 0;
    for (int c = 0; c < 3; ++c) {
        float conf[7];
        for (int i = 0; i < 7; ++i) conf[i] = (i < n_conf[c]) ? 0.95f : 0.10f;
        CHECK(cudaMemcpy(d_conf, conf, sizeof(conf), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_num_acc, 0, 4));
        CHECK(squid_adaptive_graph_launch(graph, s));
        CHECK(cudaStreamSynchronize(s));
        const uint8_t body = (uint8_t)(*h_num_acc & 0xff);
        int32_t tokens[8] = {};
        CHECK(cudaMemcpy(tokens, d_tokens, sizeof(tokens), cudaMemcpyDeviceToHost));
        const bool ok = (*h_width == want_width[c]) && (body == c + 1) && tokens[0] == anchor &&
                        tokens[want_width[c] - 1] == drafts[want_width[c] - 2];
        printf("case n_conf=%d: width=%d (want %d) body=%d (want %d) tokens[0]=%d -> %s\n", n_conf[c], *h_width,
               want_width[c], body, c + 1, tokens[0], ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    }
    CHECK(squid_adaptive_graph_destroy(graph));
    printf(failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
