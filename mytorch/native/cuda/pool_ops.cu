#include <cuda_fp16.h>

#include <cuda/std/limits>

#define POOL_OPERATION_FORWARD(name, op, init_value)                           \
  template <typename T>                                                        \
  __global__ void name##_reference(                                            \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, T *input, T *output) {                     \
    int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;   \
    int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;   \
    int rows = batch_size * in_channels;                                       \
    int cols = output_h * output_w;                                            \
    int pool_size = kernel_size_h * kernel_size_w;                             \
    int warp_size_k = 4;                                                       \
    int warp_size_i = warpSize / warp_size_k;                                  \
    int lane_id = threadIdx.x;                                                 \
    int yid = blockIdx.y * blockDim.y + threadIdx.y;                           \
    int zid = blockIdx.z * blockDim.z + threadIdx.z;                           \
    for (int i = yid * warp_size_i + (lane_id / warp_size_k); i < rows;        \
         i += blockDim.y * gridDim.y * warp_size_i) {                          \
      int in_channels_idx = i % in_channels;                                   \
      int batch_idx = i / in_channels;                                         \
      for (int j = zid; j < cols; j += blockDim.z * gridDim.z) {               \
        int output_x = j / output_w;                                           \
        int output_y = j % output_w;                                           \
        T ans = init_value();                                                  \
        for (int k = lane_id % warp_size_k; k < pool_size; k += warp_size_k) { \
          int kernel_size_w_idx = k % kernel_size_w;                           \
          int kernel_size_h_idx = k / kernel_size_w;                           \
          int input_x = output_x * stride_h + kernel_size_h_idx - padding_h;   \
          int input_y = output_y * stride_w + kernel_size_w_idx - padding_w;   \
          bool is_padding = input_x < 0 || input_y < 0 ||                      \
                            input_x >= input_h || input_y >= input_w;          \
          int input_idx =                                                      \
              ((batch_idx * in_channels + in_channels_idx) * input_h +         \
               input_x) *                                                      \
                  input_w +                                                    \
              input_y;                                                         \
          T input_value = is_padding ? (T)0 : input[input_idx];                \
          op(ans, input_value);                                                \
        }                                                                      \
        for (int offset = warp_size_k / 2; offset > 0; offset >>= 1) {         \
          op(ans, __shfl_down_sync(0xFFFFFFFF, ans, offset));                  \
        }                                                                      \
        if (lane_id % warp_size_k == 0) {                                      \
          output[i * cols + j] = ans;                                          \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" __global__ void name##_reference_fp32(                            \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, float *input, float *output) {             \
    name##_reference<float>(batch_size, in_channels, input_h, input_w,         \
                            kernel_size_h, kernel_size_w, stride_h, stride_w,  \
                            padding_h, padding_w, input, output);              \
  }                                                                            \
  extern "C" __global__ void name##_reference_fp16(                            \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, half *input, half *output) {               \
    name##_reference<half>(batch_size, in_channels, input_h, input_w,          \
                           kernel_size_h, kernel_size_w, stride_h, stride_w,   \
                           padding_h, padding_w, input, output);               \
  }

#define POOL_OPERATION_BACKWARD(name, op_backward)                             \
  template <typename T>                                                        \
  __global__ void name##_backward_reference(                                   \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, T *input, T *output, T *input_grad,        \
      T *output_grad) {                                                        \
    int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;   \
    int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;   \
    int rows = batch_size * in_channels;                                       \
    int cols = output_h * output_w;                                            \
    int pool_size = kernel_size_h * kernel_size_w;                             \
    int warp_size_k = 4;                                                       \
    int warp_size_i = warpSize / warp_size_k;                                  \
    int lane_id = threadIdx.x;                                                 \
    int yid = blockIdx.y * blockDim.y + threadIdx.y;                           \
    int zid = blockIdx.z * blockDim.z + threadIdx.z;                           \
    for (int i = yid * warp_size_i + (lane_id / warp_size_k); i < rows;        \
         i += blockDim.y * gridDim.y * warp_size_i) {                          \
      int in_channels_idx = i % in_channels;                                   \
      int batch_idx = i / in_channels;                                         \
      for (int j = zid; j < cols; j += blockDim.z * gridDim.z) {               \
        int output_x = j / output_w;                                           \
        int output_y = j % output_w;                                           \
        int output_idx =                                                       \
            ((batch_idx * in_channels + in_channels_idx) * output_h +          \
             output_x) *                                                       \
                output_w +                                                     \
            output_y;                                                          \
        for (int k = lane_id % warp_size_k; k < pool_size; k += warp_size_k) { \
          int kernel_size_w_idx = k % kernel_size_w;                           \
          int kernel_size_h_idx = k / kernel_size_w;                           \
          int input_x = output_x * stride_h + kernel_size_h_idx - padding_h;   \
          int input_y = output_y * stride_w + kernel_size_w_idx - padding_w;   \
          bool is_padding = input_x < 0 || input_y < 0 ||                      \
                            input_x >= input_h || input_y >= input_w;          \
          if (is_padding) continue;                                            \
          int input_idx =                                                      \
              ((batch_idx * in_channels + in_channels_idx) * input_h +         \
               input_x) *                                                      \
                  input_w +                                                    \
              input_y;                                                         \
          atomicAdd(input_grad + input_idx,                                    \
                    op_backward(input[input_idx], output[output_idx],          \
                                output_grad[output_idx]));                     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp32(                   \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, float *input, float *output,               \
      float *input_grad, float *output_grad) {                                 \
    name##_backward_reference<float>(batch_size, in_channels, input_h,         \
                                     input_w, kernel_size_h, kernel_size_w,    \
                                     stride_h, stride_w, padding_h, padding_w, \
                                     input, output, input_grad, output_grad);  \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp16(                   \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, half *input, half *output,                 \
      half *input_grad, half *output_grad) {                                   \
    name##_backward_reference<half>(batch_size, in_channels, input_h, input_w, \
                                    kernel_size_h, kernel_size_w, stride_h,    \
                                    stride_w, padding_h, padding_w, input,     \
                                    output, input_grad, output_grad);          \
  }

#define MAX_POOL2D(x, y) x = fmax(x, y)
#define MAX_POOL2D_INIT_VALUE() (-::cuda::std::numeric_limits<T>::max())
#define MAX_POOL2D_BACKWARD(i, o, og) (i == o ? og : (T)0)

POOL_OPERATION_FORWARD(max_pool2d, MAX_POOL2D, MAX_POOL2D_INIT_VALUE)
POOL_OPERATION_BACKWARD(max_pool2d, MAX_POOL2D_BACKWARD)