#include <cuda_fp16.h>

#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "reduce_utils.cuh"

template <typename T>
class MaxForward {
 public:
  int n, shape_n, num_reduce_axis;
  T *input, *output;
  int *shape, *reduce_axis;
  int64_t* indices;
  T* value_buffer;
  int64_t* indices_buffer;
  T value;
  int64_t index;
  const T inf = ::cuda::std::numeric_limits<T>::max();

  __device__ MaxForward(void* shared, int n, T* input, int shape_n, int* shape,
                        int num_reduce_axis, int* reduce_axis, int64_t* indices,
                        T* output)
      : n(n),
        input(input),
        shape_n(shape_n),
        shape(shape),
        num_reduce_axis(num_reduce_axis),
        reduce_axis(reduce_axis),
        output(output),
        indices(indices) {
    value_buffer = (T*)shared;
    indices_buffer = (int64_t*)((T*)shared + warpSize);
  }

  __device__ int inner() {
    int n = 1;
    for (int i = 0; i < num_reduce_axis; i++) n *= shape[reduce_axis[i]];
    return n;
  }

  __device__ int outer() {
    int n = 1;
    for (int i = 0; i < shape_n; i++) n *= shape[i];
    return n / inner();
  }

  __device__ void InitOuter() { value = -inf; }

  __device__ void LoopInner(int o, int i) {
    int idx =
        restore_reduction(shape_n, shape, num_reduce_axis, reduce_axis, o, i);
    if (input[idx] > value) {
      value = input[idx];
      index = i;
    }
  }

  __device__ void Aggregate(int offset) {
    T other_value = __shfl_xor_sync(0xFFFFFFFF, value, offset);
    int64_t other_index = __shfl_xor_sync(0xFFFFFFFF, index, offset);
    if (other_value > value) {
      value = other_value;
      index = other_index;
    }
  }

  __device__ void WriteBuffer(int warp_id) {
    value_buffer[warp_id] = value;
    indices_buffer[warp_id] = index;
  }

  __device__ void ReadBuffer(bool is_valid, int lane_id) {
    value = is_valid ? value_buffer[lane_id] : -inf;
    index = is_valid ? indices_buffer[lane_id] : 0;
  }

  __device__ void WriteAnswer(int o) {
    output[o] = value;
    indices[o] = index;
  }
};

template <typename T>
__global__ void MaxReference(int n, T* input, int shape_n, int* shape,
                             int num_reduce_axis, int* reduce_axis,
                             int64_t* indices, T* output) {
  ReduceTemplate<T, MaxForward<T>>(n, input, shape_n, shape, num_reduce_axis,
                                   reduce_axis, indices, output);
}

template <typename T>
__global__ void max_backward_reference(int n, T* input, int shape_n, int* shape,
                                       int num_reduce_axis, int* reduce_axis,
                                       int64_t* indices, T* input_grad,
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
