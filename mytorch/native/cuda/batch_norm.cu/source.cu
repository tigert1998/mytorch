#include <cuda_fp16.h>

#include <cuda/std/cstdint>

#include "reduce_utils.cuh"

template <typename T>
class MeanForward {
 public:
  int batch_size, channels, height, width;
  T *input, *output;
  T* buffer;
  T value;

  __device__ MeanForward(void* shared, int batch_size, int channels, int height,
                         int width, T* input, T* output)
      : batch_size(batch_size),
        channels(channels),
        height(height),
        width(width),
        input(input),
        output(output) {
    buffer = (T*)shared;
  }

  __device__ int inner() { return batch_size * height * width; }

  __device__ int outer() { return channels; }

  __device__ void InitOuter() { value = 0; }

  __device__ void LoopInner(int o, int i) {
    int batch_idx = i / (height * width);
    int channel_idx = o;
    int height_idx = i / width % height;
    int width_idx = i % width;
    int idx = batch_idx * channels * height * width +
              channel_idx * height * width + height_idx * width + width_idx;
    value += input[idx] / (T)inner();
  }

  __device__ void Aggregate(int offset) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }

  __device__ void WriteBuffer(int warp_id) { buffer[warp_id] = value; }

  __device__ void ReadBuffer(bool is_valid, int lane_id) {
    value = is_valid ? buffer[lane_id] : (T)0;
  }

  __device__ void WriteAnswer(int o) { output[o] = value; }
};

template <typename T>
class VarForward {
 public:
  int batch_size, channels, height, width;
  T *input, *mean, *output;
  T* buffer;
  T value;

  __device__ VarForward(void* shared, int batch_size, int channels, int height,
                        int width, T* input, T* mean, T* output)
      : batch_size(batch_size),
        channels(channels),
        height(height),
        width(width),
        input(input),
        mean(mean),
        output(output) {
    buffer = (T*)shared;
  }

  __device__ int inner() { return batch_size * height * width; }

  __device__ int outer() { return channels; }

  __device__ void InitOuter() { value = 0; }

  __device__ void LoopInner(int o, int i) {
    int batch_idx = i / (height * width);
    int channel_idx = o;
    int height_idx = i / width % height;
    int width_idx = i % width;
    int idx = batch_idx * channels * height * width +
              channel_idx * height * width + height_idx * width + width_idx;
    T minus = input[idx] - mean[channel_idx];
    value += minus * minus / (T)inner();
  }

  __device__ void Aggregate(int offset) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }

  __device__ void WriteBuffer(int warp_id) { buffer[warp_id] = value; }

  __device__ void ReadBuffer(bool is_valid, int lane_id) {
    value = is_valid ? buffer[lane_id] : (T)0;
  }

  __device__ void WriteAnswer(int o) { output[o] = value; }
};

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
__global__ void UpdateRunningStats(int n, T* old_value, T momentum,
                                   T* new_value) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize;
  int num_warps = gridDim.x * blockDim.x / warpSize;
  for (int i = warp_id * warpSize + lane_id; i < n; i += num_warps * warpSize) {
    old_value[i] = ((T)1 - momentum) * old_value[i] + momentum * new_value[i];
  }
}

template <typename T>
__global__ void batch_norm2d_reference(int batch_size, int channels, int height,
                                       int width, T* input, T* mean, T* var,
                                       T eps, T* weight, T* bias,
                                       int8_t training, T momentum,
                                       T* running_mean, T* running_var,
                                       T* output) {
  bool track_running_stats = running_mean != nullptr && running_var != nullptr;
  T *current_mean = nullptr, *current_var = nullptr;
  if (training || !track_running_stats) {
    dim3 grid = {1, 32, 1};
    dim3 block = {1024, 1, 1};
    int shared_bytes = (block.x / warpSize) * sizeof(T);
    ReduceTemplate<T, MeanForward<T>><<<grid, block, shared_bytes>>>(
        batch_size, channels, height, width, input, mean);
    ReduceTemplate<T, VarForward<T>><<<grid, block, shared_bytes>>>(
        batch_size, channels, height, width, input, mean, var);
    if (track_running_stats) {
      UpdateRunningStats<<<128, 128>>>(channels, running_mean, momentum, mean);
      UpdateRunningStats<<<128, 128>>>(channels, running_var, momentum, var);
    }
    current_mean = mean;
    current_var = var;
  } else {
    current_mean = running_mean;
    current_var = running_var;
  }

  dim3 grid = {32, 1, 1};
  dim3 block = {1024, 1, 1};
  compute_batch_norm<<<grid, block>>>(batch_size, channels, height, width,
                                      input, current_mean, current_var, eps,
                                      weight, bias, output);
}

template <typename T>
__global__ void BackNormBackwardElementwisePass(int batch_size, int channels,
                                                int height, int width, T* input,
                                                T* mean, T* var, T eps,
                                                T* weight, T* input_grad, T* g,
                                                T* h, T* output_grad) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  int inner = batch_size * height * width;
  int outer = channels;
  for (int o = blockIdx.y; o < outer; o += gridDim.y) {
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int batch_idx = i / (height * width);
      int channel_idx = o;
      int height_idx = i / width % height;
      int width_idx = i % width;
      int idx = batch_idx * channels * height * width +
                channel_idx * height * width + height_idx * width + width_idx;

      T w = weight != nullptr ? weight[channel_idx] : (T)1;
      T denom = (T)sqrt(var[channel_idx] + eps);
      T x_hat = (input[idx] - mean[channel_idx]) / denom;
      input_grad[idx] = w / denom *
                        (output_grad[idx] - g[channel_idx] / (T)inner -
                         x_hat * h[channel_idx] / (T)inner);
    }
  }
}

template <typename T>
class BatchNormBackwardReductionPass {
 public:
  T wg, bg;
  int batch_size, channels, height, width;
  T *w_buffer, *b_buffer;
  T eps;
  T *input, *mean, *var, *weight_grad, *bias_grad, *output_grad;

  __device__ BatchNormBackwardReductionPass(void* shared, int batch_size,
                                            int channels, int height, int width,
                                            T* input, T* mean, T* var, T eps,
                                            T* weight_grad, T* bias_grad,
                                            T* output_grad) {
    this->batch_size = batch_size;
    this->channels = channels;
    this->height = height;
    this->width = width;
    w_buffer = ((T*)shared) + 0 * warpSize;
    b_buffer = ((T*)shared) + 1 * warpSize;
    this->input = input;
    this->mean = mean;
    this->var = var;
    this->eps = eps;
    this->weight_grad = weight_grad;
    this->bias_grad = bias_grad;
    this->output_grad = output_grad;
  }

  __device__ int inner() { return batch_size * height * width; }

  __device__ int outer() { return channels; }

  __device__ void InitOuter() { wg = bg = 0; }

  __device__ void LoopInner(int o, int i) {
    int batch_idx = i / (height * width);
    int channel_idx = o;
    int height_idx = i / width % height;
    int width_idx = i % width;
    int idx = batch_idx * channels * height * width +
              channel_idx * height * width + height_idx * width + width_idx;

    T denom = (T)sqrt(var[channel_idx] + eps);
    T x_hat = (input[idx] - mean[channel_idx]) / denom;
    wg += output_grad[idx] * x_hat;
    bg += output_grad[idx];
  }

  __device__ void Aggregate(int offset) {
    wg += __shfl_xor_sync(0xFFFFFFFF, wg, offset);
    bg += __shfl_xor_sync(0xFFFFFFFF, bg, offset);
  }

  __device__ void WriteBuffer(int warp_id) {
    w_buffer[warp_id] = wg;
    b_buffer[warp_id] = bg;
  }

  __device__ void ReadBuffer(bool is_valid, int lane_id) {
    wg = is_valid ? w_buffer[lane_id] : (T)0;
    bg = is_valid ? b_buffer[lane_id] : (T)0;
  }

  __device__ void WriteAnswer(int o) {
    weight_grad[o] = wg;
    bias_grad[o] = bg;
  }
};

template <typename T>
__global__ void batch_norm2d_backward_reference(int batch_size, int channels,
                                                int height, int width, T* input,
                                                T* mean, T* var, T eps,
                                                T* weight, T* bias,
                                                T* input_grad, T* weight_grad,
                                                T* bias_grad, T* output_grad) {
  dim3 grid = {1, 32, 1};
  dim3 block = {1024, 1, 1};

  int shared_bytes = 2 * warpSize * sizeof(T);
  ReduceTemplate<T, BatchNormBackwardReductionPass<T>>
      <<<grid, block, shared_bytes>>>(batch_size, channels, height, width,
                                      input, mean, var, eps, weight_grad,
                                      bias_grad, output_grad);

  grid = {1, 32, 1};
  block = {1024, 1, 1};
  BackNormBackwardElementwisePass<<<grid, block>>>(
      batch_size, channels, height, width, input, mean, var, eps, weight,
      input_grad, bias_grad, weight_grad, output_grad);
}
