#include <cuda_fp16.h>

#include <cuda/std/limits>

template <typename T>
__global__ void cross_entropy_reference(int batch_size, int num_classes,
                                        T* input, int* labels, T* output) {
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  const int warps_per_block = blockDim.x / warpSize;
  const int num_warps = warps_per_block * gridDim.x;
  const int global_warp_id = warps_per_block * blockIdx.x + warp_id;

  for (int batch_idx = global_warp_id; batch_idx < batch_size;
       batch_idx += num_warps) {
    T* x = input + batch_idx * num_classes;

    // calc max value of digits
    T max_val = -::cuda::std::numeric_limits<T>::max();
    for (int i = lane_id; i < num_classes; i += warpSize) {
      max_val = fmax(max_val, x[i]);
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      max_val = fmax(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // calc denominator
    T sum_val = (T)0;
    for (int i = lane_id; i < num_classes; i += warpSize) {
      sum_val += exp(x[i] - max_val);
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, offset);
    }

    // calc loss
    T loss = (T)0;
    for (int i = lane_id; i < num_classes; i += warpSize) {
      T y = labels[batch_idx] == i ? (T)1 : (T)0;
      loss += -y * (T)log((T)exp(x[i] - max_val) / sum_val);
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      loss += __shfl_xor_sync(0xFFFFFFFF, loss, offset);
    }

    if (lane_id == 0) atomicAdd(&output[0], loss / (T)batch_size);
  }
}

extern "C" __global__ void cross_entropy_reference_fp32(
    int batch_size, int num_classes, float* input, int* labels, float* output) {
  cross_entropy_reference(batch_size, num_classes, input, labels, output);
}

extern "C" __global__ void cross_entropy_reference_fp16(
    int batch_size, int num_classes, half* input, int* labels, half* output) {
  cross_entropy_reference(batch_size, num_classes, input, labels, output);
}

template <typename T>
__global__ void cross_entropy_backward_reference(int batch_size,
                                                 int num_classes, T* input,
                                                 int* labels, T* input_grad,
                                                 T* output_grad) {
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  const int warps_per_block = blockDim.x / warpSize;
  const int num_warps = warps_per_block * gridDim.x;
  const int global_warp_id = warps_per_block * blockIdx.x + warp_id;

  for (int batch_idx = global_warp_id; batch_idx < batch_size;
       batch_idx += num_warps) {
    T* x = input + batch_idx * num_classes;

    // calc max value of digits
    T max_val = -::cuda::std::numeric_limits<T>::max();
    for (int i = lane_id; i < num_classes; i += warpSize) {
      max_val = fmax(max_val, x[i]);
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      max_val = fmax(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // calc denominator
    T sum_val = (T)0;
    for (int i = lane_id; i < num_classes; i += warpSize) {
      sum_val += exp(x[i] - max_val);
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, offset);
    }

    // calc input grad
    for (int i = lane_id; i < num_classes; i += warpSize) {
      int mem_idx = batch_idx * num_classes + i;
      T y = labels[batch_idx] == i ? (T)1 : (T)0;
      input_grad[mem_idx] = ((T)exp(x[i] - max_val) / sum_val - y) *
                            output_grad[0] / (T)batch_size;
    }
  }
}

extern "C" __global__ void cross_entropy_backward_reference_fp32(
    int batch_size, int num_classes, float* input, int* labels,
    float* input_grad, float* output_grad) {
  cross_entropy_backward_reference(batch_size, num_classes, input, labels,
                                   input_grad, output_grad);
}

extern "C" __global__ void cross_entropy_backward_reference_fp16(
    int batch_size, int num_classes, half* input, int* labels, half* input_grad,
    half* output_grad) {
  cross_entropy_backward_reference(batch_size, num_classes, input, labels,
                                   input_grad, output_grad);
}