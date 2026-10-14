// Microbenchmark: int8 GEMV variants vs. achievable memory bandwidth.
#include "../kernels/drama.cu"
#include <cstdio>
#include <vector>

template <int K, int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32)
    v_gemv(const int8_t* __restrict__ W, const __half* __restrict__ S, const float* __restrict__ x, float* y, int N) {
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
}

__global__ void read_bw(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldg(p + i);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345) out[0] = 1;
}

template <typename F>
float time_it(F f, int reps = 200) {
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i) f();
  cudaEventRecord(a);
  for (int i = 0; i < reps; ++i) f();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  return ms * 1000.f / reps;
}

static size_t g_big;
template <int K, int WARPS, int RPW>
void run(const char* name, int N, int8_t* W, __half* S, float* x, float* y) {
  const int blocks = (N + WARPS * RPW - 1) / (WARPS * RPW);
  const size_t mat = (size_t)N * K;
  const int copies = (int)(g_big / mat);
  int it = 0;
  // Rotate through distinct matrices so every launch streams from DRAM, as in decode.
  float us = time_it([&] {
    const size_t off = (size_t)(it++ % copies) * mat;
    v_gemv<K, WARPS, RPW><<<blocks, WARPS * 32>>>(W + off, S + off / 128, x, y, N);
  });
  double bytes = (double)N * K + (double)N * K / 128 * 2;
  printf("%-28s N=%5d K=%4d: %7.2f us  %6.0f GB/s\n", name, N, K, us, bytes / us / 1e3);
}

int main() {
  size_t big = 1024ull << 20;
  g_big = big;
  int8_t* W; __half* S; float *x, *y; int* o;
  cudaMalloc(&W, big); cudaMalloc(&S, big / 64); cudaMalloc(&x, 1 << 20); cudaMalloc(&y, 1 << 22); cudaMalloc(&o, 4);
  cudaMemset(W, 1, big); cudaMemset(S, 0, big / 64); cudaMemset(x, 0, 1 << 20);
  for (int blocks : {256, 512, 1024, 2048}) {
    size_t n = big / 16;
    float us = time_it([&] { read_bw<<<blocks, 256>>>((const int4*)W, n, o); }, 20);
    printf("read 1GB, %4d blocks: %8.1f us %6.0f GB/s\n", blocks, us, (double)big / us / 1e3);
  }
  run<2048, 8, 2>("w8 rpw2 (current)", 12352, W, S, x, y);
  run<2048, 8, 1>("w8 rpw1", 12352, W, S, x, y);
  run<2048, 4, 1>("w4 rpw1", 12352, W, S, x, y);
  run<2048, 4, 2>("w4 rpw2", 12352, W, S, x, y);
  run<2048, 8, 4>("w8 rpw4", 12352, W, S, x, y);
  run<2048, 16, 1>("w16 rpw1", 12352, W, S, x, y);
  run<4096, 8, 2>("w8 rpw2 (current)", 2048, W, S, x, y);
  run<4096, 8, 1>("w8 rpw1", 2048, W, S, x, y);
  run<4096, 4, 1>("w4 rpw1", 2048, W, S, x, y);
  run<4096, 2, 1>("w2 rpw1", 2048, W, S, x, y);
  run<2048, 8, 2>("w8 rpw2 (current)", 9216, W, S, x, y);
  run<2048, 4, 1>("w4 rpw1", 9216, W, S, x, y);
  run<2048, 8, 2>("w8 rpw2 (current)", 4096, W, S, x, y);
  run<2048, 4, 1>("w4 rpw1", 4096, W, S, x, y);
  run<6144, 8, 2>("w8 rpw2 (current)", 2048, W, S, x, y);
  run<6144, 4, 1>("w4 rpw1", 2048, W, S, x, y);
  run<6144, 2, 1>("w2 rpw1", 2048, W, S, x, y);
  return 0;
}
