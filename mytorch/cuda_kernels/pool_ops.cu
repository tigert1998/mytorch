#include <cuda_fp16.h>

#include <cuda/std/limits>

#define POOL_OPERATION_FORWARD(name, op)                                       \
  template <typename T>                                                        \
  __global__ void name##_reference(                                            \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, T *input, T *output) {                     \
    int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;   \
    int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;   \
    int xid = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int yid = blockIdx.y * blockDim.y + threadIdx.y;                           \
    int zid = blockIdx.z * blockDim.z + threadIdx.z;                           \
    if (xid >= batch_size * in_channels || yid >= output_h || zid >= output_w) \
      return;                                                                  \
    int batch_idx = xid / in_channels;                                         \
    int in_c = xid % in_channels;                                              \
    T ans = 0;                                                                 \
    for (int i = 0; i < kernel_size_h; i++)                                    \
      for (int j = 0; j < kernel_size_w; j++) {                                \
        int input_x = yid * stride_h + i - padding_h;                          \
        int input_y = zid * stride_w + j - padding_w;                          \
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||  \
                          input_y >= input_w;                                  \
        int input_idx =                                                        \
            ((batch_idx * in_channels + in_c) * input_h + input_x) * input_w + \
            input_y;                                                           \
        T input_value = is_padding ? (T)0 : input[input_idx];                  \
        op();                                                                  \
      }                                                                        \
    output[(xid * output_h + yid) * output_w + zid] = ans;                     \
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

#define POOL_OPERATION_BACKWARD(name, op_backward_init_hook,                   \
                                op_backward_loop_hook,                         \
                                op_backward_finalize_hook)                     \
  template <typename T>                                                        \
  __global__ void name##_backward_reference(                                   \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, T *input, T *input_grad, T *output_grad) { \
    int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;   \
    int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;   \
    int xid = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int yid = blockIdx.y * blockDim.y + threadIdx.y;                           \
    int zid = blockIdx.z * blockDim.z + threadIdx.z;                           \
    if (xid >= batch_size * in_channels || yid >= output_h || zid >= output_w) \
      return;                                                                  \
    int batch_idx = xid / in_channels;                                         \
    int in_c = xid % in_channels;                                              \
    int output_idx = (xid * output_h + yid) * output_w + zid;                  \
    op_backward_init_hook();                                                   \
    for (int i = 0; i < kernel_size_h; i++)                                    \
      for (int j = 0; j < kernel_size_w; j++) {                                \
        int input_x = yid * stride_h + i - padding_h;                          \
        int input_y = zid * stride_w + j - padding_w;                          \
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||  \
                          input_y >= input_w;                                  \
        int input_idx =                                                        \
            ((batch_idx * in_channels + in_c) * input_h + input_x) * input_w + \
            input_y;                                                           \
        T input_value = is_padding ? (T)0 : input[input_idx];                  \
        op_backward_loop_hook();                                               \
      }                                                                        \
    op_backward_finalize_hook();                                               \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp32(                   \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, float *input, float *input_grad,           \
      float *output_grad) {                                                    \
    name##_backward_reference<float>(batch_size, in_channels, input_h,         \
                                     input_w, kernel_size_h, kernel_size_w,    \
                                     stride_h, stride_w, padding_h, padding_w, \
                                     input, input_grad, output_grad);          \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp16(                   \
      int batch_size, int in_channels, int input_h, int input_w,               \
      int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,        \
      int padding_h, int padding_w, half *input, half *input_grad,             \
      half *output_grad) {                                                     \
    name##_backward_reference<half>(batch_size, in_channels, input_h, input_w, \
                                    kernel_size_h, kernel_size_w, stride_h,    \
                                    stride_w, padding_h, padding_w, input,     \
                                    input_grad, output_grad);                  \
  }

#define MAX_POOL2D() ans = fmax(ans, input_value)
#define MAX_POOL2D_BACKWARD_INIT_HOOK() \
  int dst_input_idx;                    \
  T dst_input_value = -::cuda::std::numeric_limits<T>::max();
#define MAX_POOL2D_BACKWARD_LOOP_HOOK()                       \
  do {                                                        \
    bool update = input_value > dst_input_value;              \
    dst_input_idx = update ? input_idx : dst_input_idx;       \
    dst_input_value = update ? input_value : dst_input_value; \
  } while (0)
#define MAX_POOL2D_BACKWARD_FINALIZE_HOOK() \
  atomicAdd(&input_grad[dst_input_idx], output_grad[output_idx])

POOL_OPERATION_FORWARD(max_pool2d, MAX_POOL2D)
POOL_OPERATION_BACKWARD(max_pool2d, MAX_POOL2D_BACKWARD_INIT_HOOK,
                        MAX_POOL2D_BACKWARD_LOOP_HOOK,
                        MAX_POOL2D_BACKWARD_FINALIZE_HOOK)