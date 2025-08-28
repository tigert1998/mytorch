#include <cuda_fp16.h>

template <typename T>
__global__ void compute_mean(int batch_size, int channels, int height,
                             int width, T* input, T* output) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  extern __shared__ char shared[];
  T* buffer = (T*)shared;
  int inner = batch_size * height * width;
  int outer = channels;
  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    T value = (T)0;
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int batch_idx = i / (height * width);
      int channel_idx = x;
      int height_idx = i / width % height;
      int width_idx = i % width;
      int idx =
          ((batch_idx * channels + channel_idx) * height + height_idx) * width +
          width_idx;
      value += input[idx] / (T)inner;
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    if (lane_id == 0) {
      buffer[warp_id] = value;
    }
    __syncthreads();
    if (warp_id == 0) {
      T value = lane_id * num_warps < inner ? buffer[lane_id] : (T)0;
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
      }
      if (lane_id == 0) output[x] = value;
    }
  }
}

template <typename T>
__global__ void compute_var(int batch_size, int channels, int height, int width,
                            T* input, T* mean, T* output) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  extern __shared__ char shared[];
  T* buffer = (T*)shared;
  int inner = batch_size * height * width;
  int outer = channels;
  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    T value = (T)0;
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int batch_idx = i / (height * width);
      int channel_idx = x;
      int height_idx = i / width % height;
      int width_idx = i % width;
      int idx =
          ((batch_idx * channels + channel_idx) * height + height_idx) * width +
          width_idx;
      T minus = input[idx] - mean[channel_idx];
      value += minus * minus / (T)inner;
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    if (lane_id == 0) {
      buffer[warp_id] = value;
    }
    __syncthreads();
    if (warp_id == 0) {
      T value = lane_id * num_warps < inner ? buffer[lane_id] : (T)0;
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
      }
      if (lane_id == 0) output[x] = value;
    }
  }
}

template <typename T>
__global__ void compute_batch_norm(int batch_size, int channels, int height,
                                   int width, T* input, T* mean, T* var, T eps,
                                   T* weight, T* bias, T* output) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize;
  int num_warps = gridDim.x * blockDim.x / warpSize;
  int n = batch_size * channels * height * width;
  bool has_weight_and_bias = weight != nullptr && bias != nullptr;
  for (int i = warp_id * warpSize + lane_id; i < n; i += num_warps * warpSize) {
    int channel_idx = i / (height * width) % channels;
    T value = (input[i] - mean[channel_idx]) / (T)sqrt(var[channel_idx] + eps);
    if (has_weight_and_bias)
      value = value * weight[channel_idx] + bias[channel_idx];
    output[i] = value;
  }
}

template <typename T>
__global__ void batch_norm2d_reference(int batch_size, int channels, int height,
                                       int width, T* input, T* mean, T* var,
                                       T eps, T* weight, T* bias, T* output) {
  dim3 grid = {1, 32, 1};
  dim3 block = {1024, 1, 1};
  int shared_bytes = (block.x / warpSize) * sizeof(T);
  compute_mean<<<grid, block, shared_bytes>>>(batch_size, channels, height,
                                              width, input, mean);
  compute_var<<<grid, block, shared_bytes>>>(batch_size, channels, height,
                                             width, input, mean, var);
  grid = {32, 1, 1};
  block = {1024, 1, 1};
  compute_batch_norm<<<grid, block>>>(batch_size, channels, height, width,
                                      input, mean, var, eps, weight, bias,
                                      output);
}

extern "C" __global__ void batch_norm2d_reference_fp32(
    int batch_size, int channels, int height, int width, float* input,
    float* mean, float* var, float eps, float* weight, float* bias,
    float* output) {
  batch_norm2d_reference(batch_size, channels, height, width, input, mean, var,
                         eps, weight, bias, output);
}

extern "C" __global__ void batch_norm2d_reference_fp16(
    int batch_size, int channels, int height, int width, half* input,
    half* mean, half* var, half eps, half* weight, half* bias, half* output) {
  batch_norm2d_reference(batch_size, channels, height, width, input, mean, var,
                         eps, weight, bias, output);
}

template <typename T>
__global__ void compute_mean_backward(int batch_size, int channels, int height,
                                      int width, T* input_grad, T* mean_grad) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  int inner = batch_size * height * width;
  int outer = channels;
  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int batch_idx = i / (height * width);
      int channel_idx = x;
      int height_idx = i / width % height;
      int width_idx = i % width;
      int idx =
          ((batch_idx * channels + channel_idx) * height + height_idx) * width +
          width_idx;
      input_grad[idx] += mean_grad[channel_idx] / (T)inner;
    }
  }
}

template <typename T>
__global__ void compute_var_backward(int batch_size, int channels, int height,
                                     int width, T* input, T* mean,
                                     T* input_grad, T* mean_grad, T* var_grad) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  int inner = batch_size * height * width;
  int outer = channels;

  extern __shared__ char shared[];
  T* buffer = (T*)shared;

  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    T value = (T)0;
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int batch_idx = i / (height * width);
      int channel_idx = x;
      int height_idx = i / width % height;
      int width_idx = i % width;
      int idx =
          ((batch_idx * channels + channel_idx) * height + height_idx) * width +
          width_idx;
      input_grad[idx] += (T)2 * (input[idx] - mean[channel_idx]) / (T)inner *
                         var_grad[channel_idx];
      value += (T)2 * (mean[channel_idx] - input[idx]) / (T)inner *
               var_grad[channel_idx];
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    if (lane_id == 0) {
      buffer[warp_id] = value;
    }
    __syncthreads();
    if (warp_id == 0) {
      T value = lane_id * num_warps < inner ? buffer[lane_id] : (T)0;
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
      }
      if (lane_id == 0) mean_grad[x] += value;
    }
  }
}

template <typename T>
__global__ void compute_batch_norm_backward(
    int batch_size, int channels, int height, int width, T* input, T* mean,
    T* var, T eps, T* weight, T* bias, T* input_grad, T* mean_grad, T* var_grad,
    T* weight_grad, T* bias_grad, T* output_grad) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  int inner = batch_size * height * width;
  int outer = channels;
  extern __shared__ char shared[];
  T* w_buffer = ((T*)shared) + 0 * warpSize;
  T* b_buffer = ((T*)shared) + 1 * warpSize;
  T* m_buffer = ((T*)shared) + 2 * warpSize;
  T* v_buffer = ((T*)shared) + 3 * warpSize;
  bool has_weight_and_bias = weight != nullptr && bias != nullptr;
  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    int channel_idx = x;
    T wg = 0, bg = 0, mg = 0, vg = 0;
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int batch_idx = i / (height * width);
      int height_idx = i / width % height;
      int width_idx = i % width;
      int idx =
          ((batch_idx * channels + channel_idx) * height + height_idx) * width +
          width_idx;

      T grad = output_grad[idx];
      if (has_weight_and_bias) {
        wg += grad * (input[idx] - mean[channel_idx]) /
              (T)sqrt(var[channel_idx] + eps);
        bg += grad;
        grad = grad * weight[channel_idx];
      }

      input_grad[idx] = grad / (T)sqrt(var[channel_idx] + eps);
      mg += -grad / (T)sqrt(var[channel_idx] + eps);
      vg += grad * (mean[channel_idx] - input[idx]) /
            ((T)2 * (T)pow((float)(var[channel_idx] + eps), (float)1.5));
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      wg += __shfl_xor_sync(0xFFFFFFFF, wg, offset);
      bg += __shfl_xor_sync(0xFFFFFFFF, bg, offset);
      mg += __shfl_xor_sync(0xFFFFFFFF, mg, offset);
      vg += __shfl_xor_sync(0xFFFFFFFF, vg, offset);
    }
    if (lane_id == 0) {
      w_buffer[warp_id] = wg;
      b_buffer[warp_id] = bg;
      m_buffer[warp_id] = mg;
      v_buffer[warp_id] = vg;
    }
    __syncthreads();
    if (warp_id == 0) {
      T wg = lane_id * num_warps < inner ? w_buffer[lane_id] : (T)0;
      T bg = lane_id * num_warps < inner ? b_buffer[lane_id] : (T)0;
      T mg = lane_id * num_warps < inner ? m_buffer[lane_id] : (T)0;
      T vg = lane_id * num_warps < inner ? v_buffer[lane_id] : (T)0;
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        wg += __shfl_xor_sync(0xFFFFFFFF, wg, offset);
        bg += __shfl_xor_sync(0xFFFFFFFF, bg, offset);
        mg += __shfl_xor_sync(0xFFFFFFFF, mg, offset);
        vg += __shfl_xor_sync(0xFFFFFFFF, vg, offset);
      }
      if (lane_id == 0) {
        if (has_weight_and_bias) {
          weight_grad[channel_idx] = wg;
          bias_grad[channel_idx] = bg;
        }
        mean_grad[channel_idx] = mg;
        var_grad[channel_idx] = vg;
      }
    }
  }
}

template <typename T>
__global__ void batch_norm2d_backward_reference(
    int batch_size, int channels, int height, int width, T* input, T* mean,
    T* var, T eps, T* weight, T* bias, T* input_grad, T* mean_grad, T* var_grad,
    T* weight_grad, T* bias_grad, T* output_grad) {
  dim3 grid = {1, 32, 1};
  dim3 block = {1024, 1, 1};

  int shared_bytes = 4 * warpSize * sizeof(T);
  compute_batch_norm_backward<<<grid, block, shared_bytes>>>(
      batch_size, channels, height, width, input, mean, var, eps, weight, bias,
      input_grad, mean_grad, var_grad, weight_grad, bias_grad, output_grad);

  shared_bytes = warpSize * sizeof(T);
  compute_var_backward<<<grid, block, shared_bytes>>>(
      batch_size, channels, height, width, input, mean, input_grad, mean_grad,
      var_grad);

  compute_mean_backward<<<grid, block>>>(batch_size, channels, height, width,
                                         input_grad, mean_grad);
}

extern "C" __global__ void batch_norm2d_backward_reference_fp32(
    int batch_size, int channels, int height, int width, float* input,
    float* mean, float* var, float eps, float* weight, float* bias,
    float* input_grad, float* mean_grad, float* var_grad, float* weight_grad,
    float* bias_grad, float* output_grad) {
  batch_norm2d_backward_reference(
      batch_size, channels, height, width, input, mean, var, eps, weight, bias,
      input_grad, mean_grad, var_grad, weight_grad, bias_grad, output_grad);
}

extern "C" __global__ void batch_norm2d_backward_reference_fp16(
    int batch_size, int channels, int height, int width, half* input,
    half* mean, half* var, half eps, half* weight, half* bias, half* input_grad,
    half* mean_grad, half* var_grad, half* weight_grad, half* bias_grad,
    half* output_grad) {
  batch_norm2d_backward_reference(
      batch_size, channels, height, width, input, mean, var, eps, weight, bias,
      input_grad, mean_grad, var_grad, weight_grad, bias_grad, output_grad);
}