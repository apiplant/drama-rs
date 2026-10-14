// CUDA kernels for the Fish codec decoder (time-major activations [T, C]).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>

#define API extern "C"
#define DEV __device__ __forceinline__

DEV float c_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

DEV float c_block_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = c_warp_sum(v);
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : 0.f;
  return c_warp_sum(t);
}

DEV float snake(float x, float a) {
  const float s = sinf(a * x);
  return x + s * s / (a + 1e-9f);
}

// out[t] = sum_c table[off_c + codes[t, c]]; codebook 0 has 4096 rows, the others 1024.
__global__ void k_rvq_sum(const int* __restrict__ codes, int ncb, const float* __restrict__ table, float* __restrict__ out,
                          int D, int sem_size, int res_size) {
  const int t = blockIdx.x;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int c = 0; c < ncb; ++c) {
      int code = codes[t * ncb + c];
      const int size = c == 0 ? sem_size : res_size;
      code = code < 0 ? 0 : code >= size ? size - 1 : code;
      const size_t row = c == 0 ? code : (size_t)sem_size + (size_t)(c - 1) * res_size + code;
      acc += table[row * D + d];
    }
    out[(size_t)t * D + d] = acc;
  }
}

API int dr_rvq_sum(const int* codes, int frames, int ncb, const float* table, float* out, int D, int sem_size,
                   int res_size, cudaStream_t st) {
  k_rvq_sum<<<frames, 256, 0, st>>>(codes, ncb, table, out, D, sem_size, res_size);
  return 0;
}

// Causal im2col with optional snake:
//   out[t, j*C + c] = snake(x[t*s + (s-1) - (k-1-j)*d, c])
// (stride s, dilation d; the conv is left-padded by (k-1)*d + 1 - s). x points at row 0 of
// the current rows; the rows before it must be readable (history or zeros).
__global__ void k_im2col(const float* __restrict__ x, int n, int C, int k, int d, int s,
                         const float* __restrict__ alpha, __half* __restrict__ out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)n * k * C;
  if (i >= total) return;
  const int c = i % C;
  const size_t tj = i / C;
  const int j = tj % k;
  const long t = tj / k;
  float v = x[(t * s + (s - 1) - (long)(k - 1 - j) * d) * C + c];
  if (alpha) v = snake(v, alpha[c]);
  out[i] = __float2half(v);
}

API int dr_im2col(const float* x, int n, int C, int k, int d, int s, const float* alpha, __half* out,
                  cudaStream_t st) {
  const size_t total = (size_t)n * k * C;
  k_im2col<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(x, n, C, k, d, s, alpha, out);
  return 0;
}

// One residual-VQ stage per row: z = in_w . r + in_b (8 dims), code = argmax cosine(z, codebook),
// r -= table[code] (= out_proj(codebook[code]) + bias). codebook_n is L2-normalised [n, 8].
__global__ void __launch_bounds__(256)
    k_vq_encode(float* __restrict__ r, int D, const float* __restrict__ in_w, const float* __restrict__ in_b,
                const float* __restrict__ codebook_n, int n, const float* __restrict__ table, int* __restrict__ codes,
                int code_stride) {
  __shared__ float red[32];
  __shared__ float z[8];
  __shared__ float bv[32];
  __shared__ int bi[32];
  __shared__ int best;
  float* row = r + (size_t)blockIdx.x * D;
  for (int o = 0; o < 8; ++o) {
    float acc = 0.f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) acc += in_w[o * D + c] * row[c];
    acc = c_block_sum(acc, red);
    if (threadIdx.x == 0) z[o] = acc + in_b[o];
  }
  __syncthreads();
  float v = -INFINITY;
  int vi = 0x7fffffff;
  for (int e = threadIdx.x; e < n; e += blockDim.x) {
    float dot = 0.f;
#pragma unroll
    for (int o = 0; o < 8; ++o) dot += z[o] * codebook_n[e * 8 + o];
    if (dot > v) v = dot, vi = e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, vi, o);
    if (ov > v || (ov == v && oi < vi)) v = ov, vi = oi;
  }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) bv[w] = v, bi[w] = vi;
  __syncthreads();
  if (w == 0) {
    v = lane < (int)(blockDim.x >> 5) ? bv[lane] : -INFINITY;
    vi = lane < (int)(blockDim.x >> 5) ? bi[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, o);
      const int oi = __shfl_xor_sync(0xffffffffu, vi, o);
      if (ov > v || (ov == v && oi < vi)) v = ov, vi = oi;
    }
    if (lane == 0) best = vi;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += blockDim.x) row[c] -= table[(size_t)best * D + c];
  if (threadIdx.x == 0) codes[(size_t)blockIdx.x * code_stride] = best;
}

API int dr_vq_encode(float* r, int rows, int D, const float* in_w, const float* in_b, const float* codebook_n, int n,
                     const float* table, int* codes, int code_stride, cudaStream_t st) {
  k_vq_encode<<<rows, 256, 0, st>>>(r, D, in_w, in_b, codebook_n, n, table, codes, code_stride);
  return 0;
}

// Transposed-conv input: out[t] = [snake(x[t]) | snake(x[t-1])] (or just snake(x[t]) when two == 0).
__global__ void k_tconv_gather(const float* __restrict__ x, int n, int C, int two, const float* __restrict__ alpha,
                               __half* __restrict__ out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int w = two ? 2 * C : C;
  if (i >= (size_t)n * w) return;
  const long t = i / w;
  const int cc = i % w;
  const int c = cc % C;
  const long src = cc < C ? t : t - 1;
  float v = x[src * C + c];
  if (alpha) v = snake(v, alpha[c]);
  out[i] = __float2half(v);
}

API int dr_tconv_gather(const float* x, int n, int C, int two, const float* alpha, __half* out, cudaStream_t st) {
  const size_t total = (size_t)n * (two ? 2 * C : C);
  k_tconv_gather<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(x, n, C, two, alpha, out);
  return 0;
}

// out[i] = act(y[i] + bias[i % C]) * (scale ? scale[i % C] : 1) + (res ? res[i] : 0)
// act: 0 none, 1 tanh, 2 gelu (erf). out16 (optional) gets an f16 copy.
__global__ void k_epilogue(const float* __restrict__ y, size_t n, int C, const float* __restrict__ bias,
                           const float* __restrict__ scale, const float* res, int act, float* out, __half* out16) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = i % C;
  float v = y[i] + (bias ? bias[c] : 0.f);
  if (act == 1) v = tanhf(v);
  else if (act == 2) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
  if (scale) v *= scale[c];
  if (res) v += res[i];
  if (out) out[i] = v;
  if (out16) out16[i] = __float2half(v);
}

API int dr_epilogue(const float* y, size_t n, int C, const float* bias, const float* scale, const float* res, int act,
                    float* out, __half* out16, cudaStream_t st) {
  k_epilogue<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, n, C, bias, scale, res, act, out, out16);
  return 0;
}

// Depthwise causal conv (kernel k) + bias. x points at row 0; the k-1 rows before it hold
// the history (zeros at the start of a sequence).
__global__ void k_dwconv(const float* __restrict__ x, int T, int C, int k, const float* __restrict__ w,
                         const float* __restrict__ b, float* __restrict__ out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * C) return;
  const int c = i % C;
  const long t = i / C;
  float acc = b[c];
  for (int j = 0; j < k; ++j) acc += w[c * k + j] * x[(t - (k - 1 - j)) * C + c];
  out[i] = acc;
}

API int dr_dwconv(const float* x, int T, int C, int k, const float* w, const float* b, float* out, cudaStream_t st) {
  const size_t n = (size_t)T * C;
  k_dwconv<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, T, C, k, w, b, out);
  return 0;
}

// LayerNorm (affine) per row -> f16.
__global__ void k_layernorm(const float* __restrict__ x, int C, const float* __restrict__ w, const float* __restrict__ b,
                            float eps, __half* __restrict__ out) {
  __shared__ float red[32];
  const float* xr = x + (size_t)blockIdx.x * C;
  float s = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) s += xr[c];
  const float mean = c_block_sum(s, red) / C;
  float v = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) v += (xr[c] - mean) * (xr[c] - mean);
  const float inv = rsqrtf(c_block_sum(v, red) / C + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x)
    out[(size_t)blockIdx.x * C + c] = __float2half((xr[c] - mean) * inv * w[c] + b[c]);
}

API int dr_layernorm(const float* x, int rows, int C, const float* w, const float* b, float eps, __half* out,
                     cudaStream_t st) {
  k_layernorm<<<rows, 256, 0, st>>>(x, C, w, b, eps, out);
  return 0;
}

// RMSNorm per row (plain weight) -> f32 and/or f16.
__global__ void k_rmsnorm_plain(const float* __restrict__ x, int C, const float* __restrict__ w, float eps, float* out,
                                __half* out16) {
  __shared__ float red[32];
  const float* xr = x + (size_t)blockIdx.x * C;
  float s = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) s += xr[c] * xr[c];
  const float inv = rsqrtf(c_block_sum(s, red) / C + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = xr[c] * inv * w[c];
    if (out) out[(size_t)blockIdx.x * C + c] = v;
    if (out16) out16[(size_t)blockIdx.x * C + c] = __float2half(v);
  }
}

API int dr_rmsnorm_plain(const float* x, int rows, int C, const float* w, float eps, float* out, __half* out16,
                         cudaStream_t st) {
  k_rmsnorm_plain<<<rows, 256, 0, st>>>(x, C, w, eps, out, out16);
  return 0;
}

// Split qkv [n, 3*H*64] into q [n, H*64] and the k/v caches, with interleaved rope (pairs
// 2i, 2i+1) on q and k at absolute positions t0 + row. Cache slot of position p: p % ring.
// Rope cos/sin are rounded to bf16 like the reference's precomputed table.
__global__ void k_codec_rope(const float* __restrict__ qkv, int H, const float* __restrict__ inv_freq, int t0, int ring,
                             float* q, float* kc, float* vc) {
  const int r = blockIdx.x;
  const int t = t0 + r;
  const size_t slot = (size_t)(t % ring);
  const int W = H * 64;
  for (int i = threadIdx.x; i < W / 2; i += blockDim.x) {
    const int head = i / 32, p = i % 32;
    float s, c;
    sincosf((float)t * inv_freq[p], &s, &c);
    c = __bfloat162float(__float2bfloat16(c));
    s = __bfloat162float(__float2bfloat16(s));
    const size_t base = (size_t)r * 3 * W;
    const int e = head * 64 + 2 * p;
    const float q0 = qkv[base + e], q1 = qkv[base + e + 1];
    const float k0 = qkv[base + W + e], k1 = qkv[base + W + e + 1];
    q[(size_t)r * W + e] = q0 * c - q1 * s;
    q[(size_t)r * W + e + 1] = q1 * c + q0 * s;
    kc[slot * W + e] = k0 * c - k1 * s;
    kc[slot * W + e + 1] = k1 * c + k0 * s;
    vc[slot * W + e] = qkv[base + 2 * W + e];
    vc[slot * W + e + 1] = qkv[base + 2 * W + e + 1];
  }
}

API int dr_codec_rope(const float* qkv, int n, int H, const float* inv_freq, int t0, int ring, float* q, float* kc,
                      float* vc, cudaStream_t st) {
  k_codec_rope<<<n, 256, 0, st>>>(qkv, H, inv_freq, t0, ring, q, kc, vc);
  return 0;
}

// Causal windowed attention, head_dim 64: query row r (position t = t0 + r) sees positions
// max(0, t - window + 1) .. t, read from the ring caches. One warp per (row, head); each
// lane owns 2 dims. Output f16 [n, H*64].
__global__ void k_codec_attn(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
                             int n, int H, int window, int t0, int ring, __half* __restrict__ out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n * H) return;
  const int r = warp / H, h = warp % H;
  const int t = t0 + r;
  const int W = H * 64;
  const float scale = 0.125f;  // 1/sqrt(64)
  const float q0 = q[(size_t)r * W + h * 64 + 2 * lane] * scale, q1 = q[(size_t)r * W + h * 64 + 2 * lane + 1] * scale;
  const int start = t - window + 1 < 0 ? 0 : t - window + 1;
  float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
  for (int j = start; j <= t; ++j) {
    const size_t at = (size_t)(j % ring) * W + h * 64 + 2 * lane;
    const float s = c_warp_sum(q0 * kc[at] + q1 * kc[at + 1]);
    const float mn = fmaxf(m, s);
    const float corr = expf(m - mn), p = expf(s - mn);
    l = l * corr + p;
    a0 = a0 * corr + p * vc[at];
    a1 = a1 * corr + p * vc[at + 1];
    m = mn;
  }
  out[(size_t)r * W + h * 64 + 2 * lane] = __float2half(a0 / l);
  out[(size_t)r * W + h * 64 + 2 * lane + 1] = __float2half(a1 / l);
}

API int dr_codec_attn(const float* q, const float* kc, const float* vc, int n, int H, int window, int t0, int ring,
                      __half* out, cudaStream_t st) {
  const int warps = n * H;
  k_codec_attn<<<(warps * 32 + 255) / 256, 256, 0, st>>>(q, kc, vc, n, H, window, t0, ring, out);
  return 0;
}

// y[t, j] = silu(h[t, j]) * h[t, I + j] -> f16
__global__ void k_swiglu16(const float* __restrict__ h, __half* __restrict__ y, int rows, int I) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * I) return;
  const size_t r = i / I, j = i % I;
  const float g = h[r * 2 * I + j];
  y[i] = __float2half(g / (1.f + expf(-g)) * h[r * 2 * I + I + j]);
}

API int dr_swiglu16(const float* h, __half* y, int rows, int I, cudaStream_t st) {
  const size_t n = (size_t)rows * I;
  k_swiglu16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(h, y, rows, I);
  return 0;
}

// Final layer: audio[t] = tanh(b + sum_{j,c} w[j*C + c] * snake(x[t - (k-1-j), c]))
__global__ void k_conv_out(const float* __restrict__ x, int n, int C, int k, const float* __restrict__ alpha,
                           const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  float acc = 0.f;
  for (int i = lane; i < k * C; i += 32) {
    const int j = i / C, c = i % C;
    acc += w[i] * snake(x[((long)warp - (k - 1 - j)) * C + c], alpha[c]);
  }
  acc = c_warp_sum(acc);
  if (lane == 0) out[warp] = tanhf(acc + b[0]);
}

API int dr_conv_out(const float* x, int n, int C, int k, const float* alpha, const float* w, const float* b, float* out,
                    cudaStream_t st) {
  k_conv_out<<<(n * 32 + 255) / 256, 256, 0, st>>>(x, n, C, k, alpha, w, b, out);
  return 0;
}

// C[M, N] (row-major, f32) = A[M, K] (f16) . W[N, K]^T (f16) + beta * C
API int dr_gemm_f16(void* h, const __half* A, const __half* W, float* C, int M, int N, int K, float beta) {
  const float alpha = 1.f;
  return (int)cublasGemmEx((cublasHandle_t)h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A,
                           CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
