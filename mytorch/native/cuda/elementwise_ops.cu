#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cuda/std/cstdint>

#define ELEMENTWISE_OPERATION_FORWARD(name, arg_type, arg, op)                 \
  template <typename T>                                                        \
  __global__ void name##_reference(int n, T* input, arg_type(T) T* output) {   \
    int lane_id = threadIdx.x % warpSize;                                      \
    int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize; \
    int num_warps = gridDim.x * blockDim.x / warpSize;                         \
    for (int i = warp_id * warpSize; i < n; i += num_warps * warpSize) {       \
      int xid = i + lane_id;                                                   \
      if (xid < n) op();                                                       \
    }                                                                          \
  }                                                                            \
  extern "C" __global__ void name##_reference_fp32(                            \
      int n, float* input, arg_type(float) float* output) {                    \
    name##_reference(n, input, arg() output);                                  \
  }                                                                            \
  extern "C" __global__ void name##_reference_fp16(                            \
      int n, half* input, arg_type(half) half* output) {                       \
    name##_reference(n, input, arg() output);                                  \
  }

#define ELEMENTWISE_OPERATION_BACKWARD(name, arg_type, arg, op_backward)       \
  template <typename T>                                                        \
  __global__ void name##_backward_reference(                                   \
      int n, T* input, arg_type(T) T* input_grad, T* output_grad) {            \
    int lane_id = threadIdx.x % warpSize;                                      \
    int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize; \
    int num_warps = gridDim.x * blockDim.x / warpSize;                         \
    for (int i = warp_id * warpSize; i < n; i += num_warps * warpSize) {       \
      int xid = i + lane_id;                                                   \
      if (xid < n) op_backward();                                              \
    }                                                                          \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp32(                   \
      int n, float* input, arg_type(float) float* input_grad,                  \
      float* output_grad) {                                                    \
    name##_backward_reference(n, input, arg() input_grad, output_grad);        \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp16(                   \
      int n, half* input, arg_type(half) half* input_grad,                     \
      half* output_grad) {                                                     \
    name##_backward_reference(n, input, arg() input_grad, output_grad);        \
  }

#define ELEMENTWISE_OPERATION(name, arg_type, arg, op, op_backward) \
  ELEMENTWISE_OPERATION_FORWARD(name, arg_type, arg, op)            \
  ELEMENTWISE_OPERATION_BACKWARD(name, arg_type, arg, op_backward)

#define ALPHA_ARG_TYPE(T) T alpha,
#define ALPHA_ARG() alpha,
#define NO_ARG_TYPE(T)
#define NO_ARG()

#define FILL() input[xid] = alpha
ELEMENTWISE_OPERATION_FORWARD(fill, ALPHA_ARG_TYPE, ALPHA_ARG, FILL)

#define NORMAL_ARG_TYPE(T) uint64_t seed, T mean, T stddev,
#define NORMAL_ARG() seed, mean, stddev,
#define NORMAL()                                           \
  do {                                                     \
    curandState_t state;                                   \
    curand_init(seed, xid, 0, &state);                     \
    input[xid] = (T)curand_normal(&state) * stddev + mean; \
  } while (0)
ELEMENTWISE_OPERATION_FORWARD(normal, NORMAL_ARG_TYPE, NORMAL_ARG, NORMAL)

#define UNIFORM_ARG_TYPE(T) uint64_t seed, T a, T b,
#define UNIFORM_ARG() seed, a, b,
#define UNIFORM()                                         \
  do {                                                    \
    curandState_t state;                                  \
    curand_init(seed, xid, 0, &state);                    \
    input[xid] = (T)curand_uniform(&state) * (b - a) + a; \
  } while (0)
ELEMENTWISE_OPERATION_FORWARD(uniform, UNIFORM_ARG_TYPE, UNIFORM_ARG, UNIFORM)

#define RELU() output[xid] = fmax((T)0, input[xid])
#define RELU_BACKWARD() \
  input_grad[xid] = input[xid] > (T)0 ? output_grad[xid] : (T)0
ELEMENTWISE_OPERATION(relu, NO_ARG_TYPE, NO_ARG, RELU, RELU_BACKWARD)
