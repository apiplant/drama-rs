#include "../kernels/drama.cu"
#include <cstdio>

DEV void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// GEMV that, once its rows are done, prefetches this block's share of the next weights.
template <int K, int WARPS, int RPW, bool PF>
__global__ void __launch_bounds__(WARPS * 32)
    pf_gemv(const int8_t* __restrict__ W, const __half* __restrict__ S, const float* __restrict__ x, float* y, int N,
            const char* next, size_t next_bytes) {
  __shared__ float4 xs4[K / 4];
  stage_x<K>(x, xs4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * WARPS + warp) * RPW;
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const float d = dot_q8<K>(W, S, row, xs4, lane);
    if (lane == 0) y[row] = d;
  }
  if (PF && next) {
    const size_t per = (next_bytes + gridDim.x - 1) / gridDim.x;
    const size_t b0 = blockIdx.x * per, b1 = min(next_bytes, b0 + per);
    for (size_t o = b0 + threadIdx.x * 128; o < b1; o += blockDim.x * 128) prefetch_l2(next + o);
  }
}

template <typename F>
float time_it(F f, int reps = 50) {
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 3; ++i) f();
  cudaEventRecord(a);
  for (int i = 0; i < reps; ++i) f();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return ms * 1000.f / reps;
}

template <int K, bool PF>
void chain(const char* name, int N, int count, int8_t* W, __half* S, float* x, float* y, cudaStream_t st) {
  const size_t mat = (size_t)N * K;
  const int blocks = (N + 15) / 16;
  // Capture a graph of `count` dependent GEMVs over distinct matrices.
  cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
  for (int i = 0; i < count; ++i) {
    const char* next = i + 1 < count ? (const char*)(W + (i + 1) * mat) : nullptr;
    pf_gemv<K, 8, 2, PF><<<blocks, 256, 0, st>>>(W + i * mat, S + i * mat / 128, x, y, N, next, mat);
  }
  cudaStreamEndCapture(st, &g);
  cudaGraphInstantiate(&ge, g, 0);
  float us = time_it([&] { cudaGraphLaunch(ge, st); });
  double bytes = (double)count * mat * (1 + 2.0 / 128);
  printf("%-10s %3d x [%5d x %4d]: %8.1f us  %6.0f GB/s  (%.2f us/kernel)\n", name, count, N, K, us, bytes / us / 1e3, us / count);
}

int main() {
  size_t big = 1024ull << 20;
  int8_t* W; __half* S; float *x, *y;
  cudaMalloc(&W, big); cudaMalloc(&S, big / 64); cudaMalloc(&x, 1 << 20); cudaMalloc(&y, 1 << 22);
  cudaMemset(W, 1, big); cudaMemset(S, 0, big / 64); cudaMemset(x, 0, 1 << 20);
  cudaStream_t st; cudaStreamCreate(&st);
  chain<4096, false>("plain", 2048, 100, W, S, x, y, st);
  chain<4096, true>("prefetch", 2048, 100, W, S, x, y, st);
  chain<2048, false>("plain", 12352, 30, W, S, x, y, st);
  chain<2048, true>("prefetch", 12352, 30, W, S, x, y, st);
  chain<2048, false>("plain", 2048, 200, W, S, x, y, st);
  chain<2048, true>("prefetch", 2048, 200, W, S, x, y, st);
  return 0;
}
