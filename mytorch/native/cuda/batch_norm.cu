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
    int idx =
        ((batch_idx * channels + channel_idx) * height + height_idx) * width +
        width_idx;
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
    int idx =
        ((batch_idx * channels + channel_idx) * height + height_idx) * width +
        width_idx;
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

extern "C" __global__ void batch_norm2d_reference_fp32(
    int batch_size, int channels, int height, int width, float* input,
    float* mean, float* var, float eps, float* weight, float* bias,
    int8_t training, float momentum, float* running_mean, float* running_var,
    float* output) {
  batch_norm2d_reference(batch_size, channels, height, width, input, mean, var,
                         eps, weight, bias, training, momentum, running_mean,
                         running_var, output);
}

extern "C" __global__ void batch_norm2d_reference_fp16(
    int batch_size, int channels, int height, int width, half* input,
    half* mean, half* var, half eps, half* weight, half* bias, int8_t training,
    half momentum, half* running_mean, half* running_var, half* output) {
  batch_norm2d_reference(batch_size, channels, height, width, input, mean, var,
                         eps, weight, bias, training, momentum, running_mean,
                         running_var, output);
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
class VarBackward {
 public:
  int batch_size, channels, height, width;
  T *input, *mean, *input_grad, *mean_grad, *var_grad;
  T* buffer;
  T value;

  __device__ VarBackward(void* shared, int batch_size, int channels, int height,
                         int width, T* input, T* mean, T* input_grad,
                         T* mean_grad, T* var_grad)
      : batch_size(batch_size),
        channels(channels),
        height(height),
        width(width),
        input(input),
        mean(mean),
        input_grad(input_grad),
        mean_grad(mean_grad),
        var_grad(var_grad) {
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
    int idx =
        ((batch_idx * channels + channel_idx) * height + height_idx) * width +
        width_idx;
    input_grad[idx] += (T)2 * (input[idx] - mean[channel_idx]) / (T)inner() *
                       var_grad[channel_idx];
    value += (T)2 * (mean[channel_idx] - input[idx]) / (T)inner() *
             var_grad[channel_idx];
  }

  __device__ void Aggregate(int offset) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }

  __device__ void WriteBuffer(int warp_id) { buffer[warp_id] = value; }

  __device__ void ReadBuffer(bool is_valid, int lane_id) {
    value = is_valid ? buffer[lane_id] : (T)0;
  }

  __device__ void WriteAnswer(int o) { mean_grad[o] += value; }
};

template <typename T>
class BatchNormBackward {
 public:
  T wg, bg, mg, vg;
  int batch_size, channels, height, width;
  T *w_buffer, *b_buffer, *m_buffer, *v_buffer;
  T eps;
  T *input, *mean, *var, *weight, *bias, *input_grad, *mean_grad, *var_grad,
      *weight_grad, *bias_grad, *output_grad;

  __device__ BatchNormBackward(void* shared, int batch_size, int channels,
                               int height, int width, T* input, T* mean, T* var,
                               T eps, T* weight, T* bias, T* input_grad,
                               T* mean_grad, T* var_grad, T* weight_grad,
                               T* bias_grad, T* output_grad) {
    this->batch_size = batch_size;
    this->channels = channels;
    this->height = height;
    this->width = width;
    w_buffer = ((T*)shared) + 0 * warpSize;
    b_buffer = ((T*)shared) + 1 * warpSize;
    m_buffer = ((T*)shared) + 2 * warpSize;
    v_buffer = ((T*)shared) + 3 * warpSize;
    this->input = input;
    this->mean = mean;
    this->var = var;
    this->eps = eps;
    this->weight = weight;
    this->bias = bias;
    this->input_grad = input_grad;
    this->mean_grad = mean_grad;
    this->var_grad = var_grad;
    this->weight_grad = weight_grad;
    this->bias_grad = bias_grad;
    this->output_grad = output_grad;
  }

  __device__ int inner() { return batch_size * height * width; }

  __device__ int outer() { return channels; }

  __device__ void InitOuter() { wg = bg = mg = vg = 0; }

  __device__ inline bool has_weight_and_bias() {
    return weight != nullptr && bias != nullptr;
  }

  __device__ void LoopInner(int o, int i) {
    int batch_idx = i / (height * width);
    int height_idx = i / width % height;
    int width_idx = i % width;
    int idx =
        ((batch_idx * channels + o) * height + height_idx) * width + width_idx;

    T grad = output_grad[idx];
    if (has_weight_and_bias()) {
      wg += grad * (input[idx] - mean[o]) / (T)sqrt(var[o] + eps);
      bg += grad;
      grad = grad * weight[o];
    }

    input_grad[idx] = grad / (T)sqrt(var[o] + eps);
    mg += -grad / (T)sqrt(var[o] + eps);
    vg += grad * (mean[o] - input[idx]) /
          ((T)2 * (T)pow((float)(var[o] + eps), (float)1.5));
  }

  __device__ void Aggregate(int offset) {
    wg += __shfl_xor_sync(0xFFFFFFFF, wg, offset);
    bg += __shfl_xor_sync(0xFFFFFFFF, bg, offset);
    mg += __shfl_xor_sync(0xFFFFFFFF, mg, offset);
    vg += __shfl_xor_sync(0xFFFFFFFF, vg, offset);
  }

  __device__ void WriteBuffer(int warp_id) {
    w_buffer[warp_id] = wg;
    b_buffer[warp_id] = bg;
    m_buffer[warp_id] = mg;
    v_buffer[warp_id] = vg;
  }

  __device__ void ReadBuffer(bool is_valid, int lane_id) {
    wg = is_valid ? w_buffer[lane_id] : (T)0;
    bg = is_valid ? b_buffer[lane_id] : (T)0;
    mg = is_valid ? m_buffer[lane_id] : (T)0;
    vg = is_valid ? v_buffer[lane_id] : (T)0;
  }

  __device__ void WriteAnswer(int o) {
    if (has_weight_and_bias()) {
      weight_grad[o] = wg;
      bias_grad[o] = bg;
    }
    mean_grad[o] = mg;
    var_grad[o] = vg;
  }
};

template <typename T>
__global__ void batch_norm2d_backward_reference(
    int batch_size, int channels, int height, int width, T* input, T* mean,
    T* var, T eps, T* weight, T* bias, T* input_grad, T* mean_grad, T* var_grad,
    T* weight_grad, T* bias_grad, T* output_grad) {
  dim3 grid = {1, 32, 1};
  dim3 block = {1024, 1, 1};

  int shared_bytes = 4 * warpSize * sizeof(T);
  ReduceTemplate<T, BatchNormBackward<T>><<<grid, block, shared_bytes>>>(
      batch_size, channels, height, width, input, mean, var, eps, weight, bias,
      input_grad, mean_grad, var_grad, weight_grad, bias_grad, output_grad);

  shared_bytes = warpSize * sizeof(T);
  ReduceTemplate<T, VarBackward<T>><<<grid, block, shared_bytes>>>(
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