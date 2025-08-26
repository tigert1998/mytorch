#include <cuda_fp16.h>

#include <cuda/std/limits>

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

template <typename T>
__global__ void max_backward_reference(int n, T* input, int shape_n, int* shape,
                                       int num_reduce_axis, int* reduce_axis,
                                       long long* indices, T* input_grad,
                                       T* output_grad) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;

  int inner = 1;
  for (int i = 0; i < num_reduce_axis; i++) inner *= shape[reduce_axis[i]];
  int outer = n / inner;

  for (int x = blockIdx.y; x < outer; x += gridDim.y) {
    for (int i = warp_id * warpSize + lane_id; i < inner;
         i += num_warps * warpSize) {
      int idx =
          restore_reduction(shape_n, shape, num_reduce_axis, reduce_axis, x, i);
      input_grad[idx] = i == indices[x] ? output_grad[x] : (T)0;
    }
  }
}

extern "C" __global__ void max_backward_reference_fp32(
    int n, float* input, int shape_n, int* shape, int num_reduce_axis,
    int* reduce_axis, long long* indices, float* input_grad,
    float* output_grad) {
  max_backward_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis,
                         indices, input_grad, output_grad);
}

extern "C" __global__ void max_backward_reference_fp16(
    int n, half* input, int shape_n, int* shape, int num_reduce_axis,
    int* reduce_axis, long long* indices, half* input_grad, half* output_grad) {
  max_backward_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis,
                         indices, input_grad, output_grad);
}