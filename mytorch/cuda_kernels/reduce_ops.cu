#include <cuda_fp16.h>

__device__ int reduce_shape(int xid, int shape_n, int* shape,
                            int num_reduce_axis, int* reduce_axis) {
  int tmp = xid, mul = 1, to = 0;
  for (int i = shape_n - 1, j = num_reduce_axis - 1; i >= 0; i--) {
    int dim = tmp % shape[i];
    tmp /= shape[i];
    int cur_shape = shape[i];

    if (j >= 0 && i == reduce_axis[j]) {
      dim = 0;
      j--;
      cur_shape = 1;
    }
    to += mul * dim;

    mul *= cur_shape;
  }
  return to;
}

#define REDUCE_OPERATION_FORWARD(name, arg_type, arg, op)                     \
  template <typename T>                                                       \
  __global__ void name##_reference(int n, T* input, arg_type(T) int shape_n,  \
                                   int* shape, int num_reduce_axis,           \
                                   int* reduce_axis, T* output) {             \
    int xid = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (xid >= n) return;                                                     \
    int to = reduce_shape(xid, shape_n, shape, num_reduce_axis, reduce_axis); \
    op();                                                                     \
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
    int xid = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (xid >= n) return;                                                      \
    int to = reduce_shape(xid, shape_n, shape, num_reduce_axis, reduce_axis);  \
    op_backward();                                                             \
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

#define REDUCE_OPERATION(name, arg_type, arg, op, op_backward) \
  REDUCE_OPERATION_FORWARD(name, arg_type, arg, op)            \
  REDUCE_OPERATION_BACKWARD(name, arg_type, arg, op_backward)

#define SCALE_ARG_TYPE(T) T scale,
#define SCALE_ARG() scale,
#define NO_ARG_TYPE(T)
#define NO_ARG()

#define SUM_SCALE() atomicAdd(&output[to], input[xid] * scale)
#define SUM_SCALE_BACKWARD() input_grad[xid] = output_grad[to] * scale
REDUCE_OPERATION(sum_scale, SCALE_ARG_TYPE, SCALE_ARG, SUM_SCALE,
                 SUM_SCALE_BACKWARD)