#include <cuda_fp16.h>

#include <cuda/std/limits>

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

#define REDUCE_OPERATION_FORWARD(name, arg_type, arg, op)                      \
  template <typename T>                                                        \
  __global__ void name##_reference(int n, T* input, arg_type(T) int shape_n,   \
                                   int* shape, int num_reduce_axis,            \
                                   int* reduce_axis, T* output) {              \
    int lane_id = threadIdx.x % warpSize;                                      \
    int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize; \
    int num_warps = gridDim.x * blockDim.x / warpSize;                         \
    for (int i = warp_id * warpSize; i < n; i += num_warps * warpSize) {       \
      int xid = i + lane_id;                                                   \
      if (xid < n) {                                                           \
        int to =                                                               \
            reduce_shape(xid, shape_n, shape, num_reduce_axis, reduce_axis);   \
        op();                                                                  \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  extern "C" __global__ void name##_reference_fp32(                            \
      int n, float* input, arg_type(float) int shape_n, int* shape,            \
      int num_reduce_axis, int* reduce_axis, float* output) {                  \
    name##_reference(n, input, arg() shape_n, shape, num_reduce_axis,          \
                     reduce_axis, output);                                     \
  }                                                                            \
  extern "C" __global__ void name##_reference_fp16(                            \
      int n, half* input, arg_type(half) int shape_n, int* shape,              \
      int num_reduce_axis, int* reduce_axis, half* output) {                   \
    name##_reference(n, input, arg() shape_n, shape, num_reduce_axis,          \
                     reduce_axis, output);                                     \
  }

#define REDUCE_OPERATION_BACKWARD(name, arg_type, arg, op_backward)            \
  template <typename T>                                                        \
  __global__ void name##_backward_reference(                                   \
      int n, T* input, arg_type(T) int shape_n, int* shape,                    \
      int num_reduce_axis, int* reduce_axis, T* input_grad, T* output_grad) {  \
    int lane_id = threadIdx.x % warpSize;                                      \
    int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize; \
    int num_warps = gridDim.x * blockDim.x / warpSize;                         \
    for (int i = warp_id * warpSize; i < n; i += num_warps * warpSize) {       \
      int xid = i + lane_id;                                                   \
      if (xid < n) {                                                           \
        int to =                                                               \
            reduce_shape(xid, shape_n, shape, num_reduce_axis, reduce_axis);   \
        op_backward();                                                         \
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

__device__ int restore_reduction(int shape_n, int* shape, int num_reduce_axis,
                                 int* reduce_axis, int outer_idx,
                                 int inner_idx) {
  int mul = 1, dest = 0;

  for (int i = shape_n - 1, j = num_reduce_axis - 1; i >= 0; i--) {
    int cur_shape = shape[i];
    if (j >= 0 && i == reduce_axis[j]) {
      dest += (inner_idx % shape[i]) * mul;
      inner_idx /= shape[i];
      j--;
    } else {
      dest += (outer_idx % shape[i]) * mul;
      outer_idx /= shape[i];
    }
    mul *= shape[i];
  }

  return dest;
}

template <typename T>
__global__ void max_reference(int n, T* input, int shape_n, int* shape,
                              int num_reduce_axis, int* reduce_axis,
                              long long* indices, T* output) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;

  extern __shared__ char shared[];
  T* value_buffer = (T*)shared;
  long long* indices_buffer = (long long*)(shared + sizeof(T) * num_warps);

  int inner = 1;
  for (int i = 0; i < num_reduce_axis; i++) inner *= shape[reduce_axis[i]];
  int outer = n / inner;

  T inf = ::cuda::std::numeric_limits<T>::max();

  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    T max_value = -inf;
    long long max_index;
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int idx =
          restore_reduction(shape_n, shape, num_reduce_axis, reduce_axis, x, i);
      if (input[idx] > max_value) {
        max_value = input[idx];
        max_index = i;
      }
    }

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      T other_max_value = __shfl_xor_sync(0xFFFFFFFF, max_value, offset);
      long long other_max_index =
          __shfl_xor_sync(0xFFFFFFFF, max_index, offset);
      if (other_max_value > max_value) {
        max_value = other_max_value;
        max_index = other_max_index;
      }
    }

    if (lane_id == 0) {
      value_buffer[warp_id] = max_value;
      indices_buffer[warp_id] = max_index;
    }
    __syncthreads();

    if (warp_id == 0) {
      T max_value = lane_id < num_warps ? value_buffer[lane_id] : -inf;
      long long max_index = lane_id < num_warps ? indices_buffer[lane_id] : 0;
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        T other_max_value = __shfl_xor_sync(0xFFFFFFFF, max_value, offset);
        long long other_max_index =
            __shfl_xor_sync(0xFFFFFFFF, max_index, offset);
        if (other_max_value > max_value) {
          max_value = other_max_value;
          max_index = other_max_index;
        }
      }

      if (lane_id == 0) {
        output[x] = max_value;
        indices[x] = max_index;
      }
    }
  }
}

extern "C" __global__ void max_reference_fp32(int n, float* input, int shape_n,
                                              int* shape, int num_reduce_axis,
                                              int* reduce_axis,
                                              long long* indices,
                                              float* output) {
  max_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis, indices,
                output);
}

extern "C" __global__ void max_reference_fp16(int n, half* input, int shape_n,
                                              int* shape, int num_reduce_axis,
                                              int* reduce_axis,
                                              long long* indices,
                                              half* output) {
  max_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis, indices,
                output);
}