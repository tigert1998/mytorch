#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/cstdint>
#include <memory>
#include <string>
#include <vector>

class ColMajorLayout {
 public:
  int height, width;
  __device__ __host__ ColMajorLayout(int height, int width)
      : height(height), width(width) {}
  __device__ __host__ int Index(int x, int y) { return x + y * height; }
};

class RowMajorLayout {
 public:
  int height, width;
  __device__ __host__ RowMajorLayout(int height, int width)
      : height(height), width(width) {}
  __device__ __host__ int Index(int x, int y) { return x * width + y; }
};

template <typename T, typename L>
class MatrixWrapper {
 public:
  T *ptr;
  L layout;
  __device__ __host__ MatrixWrapper(T *ptr, L layout)
      : ptr(ptr), layout(layout) {}
  __device__ __host__ void SetNoCheck(int x, int y, T value) {
    ptr[layout.Index(x, y)] = value;
  }
  __device__ __host__ void Set(int x, int y, T value) {
    int i = layout.Index(x, y);
    if (i >= 0) ptr[i] = value;
  }
  __device__ __host__ void Add(int x, int y, T value) {
    int i = layout.Index(x, y);
    if (i >= 0) atomicAdd(ptr + i, value);
  }
  __device__ __host__ T Get(int x, int y) {
    int i = layout.Index(x, y);
    return i < 0 ? (T)0 : ptr[i];
  }
};

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, typename M1, typename M2>
__forceinline__ __device__ void StoreIntoSMEM(
    int i, int warp_id, int lane_id, int num_warps, M1 d_a, M2 d_b,
    MatrixWrapper<T, ColMajorLayout> s_a,
    MatrixWrapper<T, RowMajorLayout> s_b) {
  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;

#pragma unroll
  for (int j = warp_id * warpSize + lane_id;
       j < max(block_size_m, block_size_n) * block_size_k;
       j += num_warps * warpSize) {
    if (j < block_size_m * block_size_k) {
      int x = j % block_size_m, y = j / block_size_m;
      s_a.SetNoCheck(x, y, d_a.Get(offset_m + x, (block_size_k * (i)) + y));
    }
    if (j < block_size_n * block_size_k) {
      int x = j / block_size_n, y = j % block_size_n;
      s_b.SetNoCheck(x, y, d_b.Get((block_size_k * (i)) + x, offset_n + y));
    }
  }
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t thread_size_m, uint32_t thread_size_n>
__forceinline__ __device__ void LoadFromSMEM(T *l, T *r, T *s_a_mem, T *s_b_mem,
                                             int tx, int ty, int j) {
#pragma unroll
  for (int k = 0; k < thread_size_m; k++)
    l[k] = *(s_a_mem + (j)*block_size_m + tx * thread_size_m + k);
#pragma unroll
  for (int k = 0; k < thread_size_n; k++)
    r[k] = *(s_b_mem + (j)*block_size_n + ty * thread_size_n + k);
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t thread_size_m, uint32_t thread_size_n>
__forceinline__ __device__ void ComputeRegisters(T *accum, T *l, T *r) {
#pragma unroll
  for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
    accum[idx] += l[idx % thread_size_m] * r[idx / thread_size_m];
  }
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, uint32_t thread_size_m, uint32_t thread_size_n>
__forceinline__ __device__ void ComputePrefetch(T *accum, int tx, int ty,
                                                T *s_a_mem, T *s_b_mem) {
  T l[2][thread_size_m], r[2][thread_size_n];
  LoadFromSMEM<T, block_size_m, block_size_n, thread_size_m, thread_size_n>(
      l[0], r[0], s_a_mem, s_b_mem, tx, ty, 0);
  for (int j = 0; j < block_size_k - 1; j++) {
    LoadFromSMEM<T, block_size_m, block_size_n, thread_size_m, thread_size_n>(
        l[(j + 1) % 2], r[(j + 1) % 2], s_a_mem, s_b_mem, tx, ty, j + 1);
    ComputeRegisters<T, block_size_m, block_size_n, thread_size_m,
                     thread_size_n>(accum, l[j % 2], r[j % 2]);
  }
  ComputeRegisters<T, block_size_m, block_size_n, thread_size_m, thread_size_n>(
      accum, l[(block_size_k - 1) % 2], r[(block_size_k - 1) % 2]);
}

template <typename T, uint32_t block_size_m, uint32_t block_size_n,
          uint32_t block_size_k, typename M1, typename M2, typename M3>
__global__ void MatrixMulGeneral(uint32_t m, uint32_t n, uint32_t k, M1 d_a,
                                 M2 d_b, M3 d_c, T *bias) {
  static constexpr uint32_t thread_size_m = block_size_m / 16;
  static constexpr uint32_t thread_size_n = block_size_n / 16;

  extern __shared__ char shared[];

  static constexpr uint32_t num_warps = 8;

  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  // z-order
  // chinese doc: https://zhuanlan.zhihu.com/p/690052715
  int tx = (warp_id / 2) * 4 + (lane_id % 8) / 2;
  int ty = (warp_id % 2) * 8 + (lane_id / 8) * 2 + lane_id % 2;

  int num_subs = (k + block_size_k - 1) / block_size_k;

  T accum[thread_size_m * thread_size_n] = {(T)0};

  T *s_a_mem = (T *)shared;
  T *s_b_mem = (T *)shared + block_size_m * block_size_k;
  auto s_a = MatrixWrapper<T, ColMajorLayout>{
      s_a_mem, ColMajorLayout(block_size_m, block_size_k)};
  auto s_b = MatrixWrapper<T, RowMajorLayout>{
      s_b_mem, RowMajorLayout(block_size_k, block_size_n)};
  for (int i = 0; i < num_subs; i++) {
    __syncthreads();
    StoreIntoSMEM<T, block_size_m, block_size_n, block_size_k>(
        i, warp_id, lane_id, num_warps, d_a, d_b, s_a, s_b);
    __syncthreads();
    ComputePrefetch<T, block_size_m, block_size_n, block_size_k, thread_size_m,
                    thread_size_n>(accum, tx, ty, s_a_mem, s_b_mem);
  }

  int offset_m = block_size_m * blockIdx.x;
  int offset_n = block_size_n * blockIdx.y;

#pragma unroll
  for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
    int x = offset_m + tx * thread_size_m + idx % thread_size_m;
    int y = offset_n + ty * thread_size_n + idx / thread_size_m;
    d_c.Set(x, y, (T)0);
  }

  __syncthreads();

#pragma unroll
  for (int idx = 0; idx < thread_size_m * thread_size_n; idx++) {
    int x = offset_m + tx * thread_size_m + idx % thread_size_m;
    int y = offset_n + ty * thread_size_n + idx / thread_size_m;
    d_c.Add(x, y, accum[idx] + (bias != nullptr ? bias[y] : 0));
  }
}

template <typename T, uint32_t R, typename M1, typename M2, typename M3>
__global__ void MatrixMulTallAndSkinny(uint32_t m, uint32_t n, uint32_t k,
                                       M1 d_a, M2 d_b, M3 d_c, T *bias) {
  extern __shared__ char shared[];
  T *sh_sum = (T *)shared;
  int stride = blockDim.x * gridDim.x;

  T local_sum[R] = {(T)0};
  for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < k; p += stride) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        local_sum[i * n + j] += d_a.Get(i, p) * d_b.Get(p, j);
      }
    }
  }

  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        sh_sum[i * n + j] = 0;
        d_c.Set(i, j, 0);
      }
    }
  }
  __syncthreads();

  for (int i = 0; i < m * n; i++) {
    atomicAdd(&sh_sum[i], local_sum[i]);
  }
  __syncthreads();

  if (tid == 0) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        d_c.Add(i, j, sh_sum[i * n + j] + (bias != nullptr ? bias[j] : 0));
      }
    }
  }
}

template <typename T, typename M1, typename M2, typename M3>
void MatrixMul(M1 a, M2 b, M3 c, T *bias, cudaStream_t stream) {
  uint32_t m = c.layout.matrix_height;
  uint32_t n = c.layout.matrix_width;
  uint32_t k = a.layout.matrix_width;

  if (m * n <= 1024) {
    dim3 block = {32, 1, 1};
    dim3 grid = {256, 1, 1};
    int shared_bytes = m * n * sizeof(T);
    MatrixMulTallAndSkinny<T, 1024>
        <<<grid, block, shared_bytes, stream>>>(m, n, k, a, b, c, bias);
  } else {
    static constexpr uint32_t block_size_m = 64, block_size_n = 64,
                              block_size_k = 8;
    constexpr int num_warps = 8;
    dim3 block = {32 * num_warps, 1, 1};
    dim3 grid = {(m + block_size_m - 1) / block_size_m,
                 (n + block_size_n - 1) / block_size_n, 1};
    int shared_bytes = (block_size_m + block_size_n) * block_size_k * sizeof(T);

    MatrixMulGeneral<T, block_size_m, block_size_n, block_size_k>
        <<<grid, block, shared_bytes, stream>>>(m, n, k, a, b, c, bias);
  }
}

class Im2colLayout {
 public:
  int channels, height, width, kernel_height, kernel_width, stride_height,
      stride_width, padding_height, padding_width;
  int output_height, output_width, matrix_height, matrix_width;
  bool t;
  __device__ __host__ Im2colLayout(int batch_size, int channels, int height,
                                   int width, int kernel_height,
                                   int kernel_width, int stride_height,
                                   int stride_width, int padding_height,
                                   int padding_width, bool t)
      : channels(channels),
        height(height),
        width(width),
        kernel_height(kernel_height),
        kernel_width(kernel_width),
        stride_height(stride_height),
        stride_width(stride_width),
        padding_height(padding_height),
        padding_width(padding_width),
        t(t) {
    output_height =
        (height + 2 * padding_height - kernel_height) / stride_height + 1;
    output_width =
        (width + 2 * padding_width - kernel_width) / stride_width + 1;

    if (t) {
      matrix_height = channels * kernel_height * kernel_width;
      matrix_width = batch_size * output_height * output_width;
    } else {
      matrix_height = batch_size * output_height * output_width;
      matrix_width = channels * kernel_height * kernel_width;
    }
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;

    int batch_idx, output_x, output_y, channel_idx, kernel_x, kernel_y;
    if (t) {
      batch_idx = y / (output_height * output_width);
      output_x = y / output_width % output_height;
      output_y = y % output_width;
      channel_idx = x / (kernel_height * kernel_width);
      kernel_x = x / kernel_width % kernel_height;
      kernel_y = x % kernel_width;
    } else {
      batch_idx = x / (output_height * output_width);
      output_x = x / output_width % output_height;
      output_y = x % output_width;
      channel_idx = y / (kernel_height * kernel_width);
      kernel_x = y / kernel_width % kernel_height;
      kernel_y = y % kernel_width;
    };

    int input_x = output_x * stride_height - padding_height + kernel_x;
    int input_y = output_y * stride_width - padding_width + kernel_y;

    if (input_x < 0 || input_x >= height || input_y < 0 || input_y >= width)
      return -1;

    return batch_idx * channels * height * width +
           channel_idx * height * width + input_x * width + input_y;
  }
};

class WeightLayout {
 public:
  int matrix_height, matrix_width;
  bool t;

  __device__ __host__ WeightLayout(int output_channels, int input_channels,
                                   int kernel_height, int kernel_width, bool t)
      : t(t) {
    if (t) {
      matrix_height = output_channels;
      matrix_width = input_channels * kernel_height * kernel_width;
    } else {
      matrix_height = input_channels * kernel_height * kernel_width;
      matrix_width = output_channels;
    }
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;
    if (t) {
      return x * matrix_width + y;
    } else {
      return y * matrix_height + x;
    }
  }
};

class OutputLayout {
 public:
  int channels, height, width;
  int matrix_height, matrix_width;
  __device__ __host__ OutputLayout(int batch_size, int channels, int height,
                                   int width)
      : channels(channels), height(height), width(width) {
    matrix_height = batch_size * height * width;
    matrix_width = channels;
  }

  __device__ __host__ int Index(int x, int y) {
    if (x < 0 || x >= matrix_height || y < 0 || y >= matrix_width) return -1;

    int channel_idx = y;
    int batch_idx = x / (height * width);
    int height_width_idx = x % (height * width);
    return batch_idx * channels * height * width +
           channel_idx * height * width + height_width_idx;
  }
};

template <typename T>
void ConvForwardImplicitGEMM(int batch_size, int input_channels, int height,
                             int width, int kernel_height, int kernel_width,
                             int stride_height, int stride_width,
                             int padding_height, int padding_width,
                             int output_channels, T *input, T *weight, T *bias,
                             T *output, cudaStream_t stream) {
  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;

  Im2colLayout im2col_layout(
      batch_size, input_channels, height, width, kernel_height, kernel_width,
      stride_height, stride_width, padding_height, padding_width, false);
  MatrixWrapper<T, Im2colLayout> a(input, im2col_layout);
  WeightLayout weight_layout(output_channels, input_channels, kernel_height,
                             kernel_width, false);
  MatrixWrapper<T, WeightLayout> b(weight, weight_layout);
  OutputLayout output_layout(batch_size, output_channels, output_height,
                             output_width);
  MatrixWrapper<T, OutputLayout> c(output, output_layout);

  MatrixMul<T>(a, b, c, bias, stream);
}

template <typename T>
void ConvBackwardImplicitGEMM(int batch_size, int input_channels, int height,
                              int width, int kernel_height, int kernel_width,
                              int stride_height, int stride_width,
                              int padding_height, int padding_width,
                              int output_channels, T *input, T *weight,
                              T *output_grad, T *input_grad, T *weight_grad,
                              cudaStream_t stream) {
  int output_height =
      (height + 2 * padding_height - kernel_height) / stride_height + 1;
  int output_width =
      (width + 2 * padding_width - kernel_width) / stride_width + 1;

  // tensors
  OutputLayout output_layout(batch_size, output_channels, output_height,
                             output_width);
  MatrixWrapper<T, OutputLayout> c(output_grad, output_layout);
  WeightLayout weight_layout(output_channels, input_channels, kernel_height,
                             kernel_width, true);
  MatrixWrapper<T, WeightLayout> b(weight, weight_layout);
  Im2colLayout im2col_layout(batch_size, input_channels, height, width,
                             kernel_height, kernel_width, stride_height,
                             stride_width, padding_height, padding_width, true);
  MatrixWrapper<T, Im2colLayout> a(input, im2col_layout);

  // grads
  Im2colLayout input_grad_layout(
      batch_size, input_channels, height, width, kernel_height, kernel_width,
      stride_height, stride_width, padding_height, padding_width, false);
  MatrixWrapper<T, Im2colLayout> input_grad_matrix(input_grad,
                                                   input_grad_layout);
  WeightLayout weight_grad_layout(output_channels, input_channels,
                                  kernel_height, kernel_width, false);
  MatrixWrapper<T, WeightLayout> weight_grad_matrix(weight_grad,
                                                    weight_grad_layout);

  // calculation
  MatrixMul<T>(c, b, input_grad_matrix, nullptr, stream);
  MatrixMul<T>(a, c, weight_grad_matrix, nullptr, stream);
}