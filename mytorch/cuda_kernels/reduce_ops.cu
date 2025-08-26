#include <cuda_fp16.h>

#include <cuda/std/limits>

#include "reduce_utils.cuh"

#define REDUCE_OPERATION_FORWARD(name, arg_type, arg, init, op, final)        \
  template <typename T>                                                       \
  __global__ void name##_reference(int n, T* input, arg_type(T) int shape_n,  \
                                   int* shape, int num_reduce_axis,           \
                                   int* reduce_axis, T* output) {             \
    int lane_id = threadIdx.x % warpSize;                                     \
    int warp_id = threadIdx.x / warpSize;                                     \
    const int num_warps = blockDim.x / warpSize;                              \
    extern __shared__ char shared[];                                          \
    T* buffer = (T*)shared;                                                   \
    int inner = 1;                                                            \
    for (int i = 0; i < num_reduce_axis; i++) inner *= shape[reduce_axis[i]]; \
    int outer = n / inner;                                                    \
    for (int x = blockIdx.y; x < outer; x += gridDim.y) {                     \
      T value = init();                                                       \
      for (int i = warp_id * warpSize + lane_id; i < inner;                   \
           i += num_warps * warpSize) {                                       \
        int idx = restore_reduction(shape_n, shape, num_reduce_axis,          \
                                    reduce_axis, x, i);                       \
        op(value, input[idx]);                                                \
      }                                                                       \
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {             \
        T other_value = __shfl_xor_sync(0xFFFFFFFF, value, offset);           \
        op(value, other_value);                                               \
      }                                                                       \
      if (lane_id == 0) {                                                     \
        buffer[warp_id] = value;                                              \
      }                                                                       \
      __syncthreads();                                                        \
      if (warp_id == 0) {                                                     \
        T value = lane_id < num_warps ? buffer[lane_id] : init();             \
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {           \
          T other_value = __shfl_xor_sync(0xFFFFFFFF, value, offset);         \
          op(value, other_value);                                             \
        }                                                                     \
        if (lane_id == 0) {                                                   \
          output[x] = final(value);                                           \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  extern "C" __global__ void name##_reference_fp32(                           \
      int n, float* input, arg_type(float) int shape_n, int* shape,           \
      int num_reduce_axis, int* reduce_axis, float* output) {                 \
    name##_reference(n, input, arg() shape_n, shape, num_reduce_axis,         \
                     reduce_axis, output);                                    \
  }                                                                           \
  extern "C" __global__ void name##_reference_fp16(                           \
      int n, half* input, arg_type(half) int shape_n, int* shape,             \
      int num_reduce_axis, int* reduce_axis, half* output) {                  \
    name##_reference(n, input, arg() shape_n, shape, num_reduce_axis,         \
                     reduce_axis, output);                                    \
  }

#define REDUCE_OPERATION_BACKWARD(name, arg_type, arg, op_backward)            \
  template <typename T>                                                        \
  __global__ void name##_backward_reference(                                   \
      int n, T* input, arg_type(T) int shape_n, int* shape,                    \
      int num_reduce_axis, int* reduce_axis, T* input_grad, T* output_grad) {  \
    int lane_id = threadIdx.x % warpSize;                                      \
    int warp_id = threadIdx.x / warpSize;                                      \
    const int num_warps = blockDim.x / warpSize;                               \
    int inner = 1;                                                             \
    for (int i = 0; i < num_reduce_axis; i++) inner *= shape[reduce_axis[i]];  \
    int outer = n / inner;                                                     \
    for (int x = blockIdx.y; x < outer; x += gridDim.y) {                      \
      for (int i = warp_id * warpSize + lane_id; i < inner;                    \
           i += num_warps * warpSize) {                                        \
        int idx = restore_reduction(shape_n, shape, num_reduce_axis,           \
                                    reduce_axis, x, i);                        \
        op_backward(input_grad[idx], output_grad[x]);                          \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp32(                   \
      int n, float* input, arg_type(float) int shape_n, int* shape,            \
      int num_reduce_axis, int* reduce_axis, float* input_grad,                \
      float* output_grad) {                                                    \
    name##_backward_reference(n, input, arg() shape_n, shape, num_reduce_axis, \
                              reduce_axis, input_grad, output_grad);           \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp16(                   \
      int n, half* input, arg_type(half) int shape_n, int* shape,              \
      int num_reduce_axis, int* reduce_axis, half* input_grad,                 \
      half* output_grad) {                                                     \
    name##_backward_reference(n, input, arg() shape_n, shape, num_reduce_axis, \
                              reduce_axis, input_grad, output_grad);           \
  }

#define SCALE_ARG_TYPE(T) T scale,
#define SCALE_ARG() scale,
#define NO_ARG_TYPE(T)
#define NO_ARG()

#define SUM_SCALE_INIT() (T)0
#define SUM_SCALE(x, y) x += y
#define SUM_SCALE_FINAL(x) (x * scale)
#define SUM_SCALE_BACKWARD(ig, og) ig = og * scale

REDUCE_OPERATION_FORWARD(sum_scale, SCALE_ARG_TYPE, SCALE_ARG, SUM_SCALE_INIT,
                         SUM_SCALE, SUM_SCALE_FINAL)
REDUCE_OPERATION_BACKWARD(sum_scale, SCALE_ARG_TYPE, SCALE_ARG,
                          SUM_SCALE_BACKWARD)