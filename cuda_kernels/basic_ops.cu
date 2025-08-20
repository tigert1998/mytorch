#include <cuda_fp16.h>

template <typename T>
__global__ void sum_reference(int n, T* input, T* output) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  atomicAdd(output, input[xid]);
}

extern "C" __global__ void sum_reference_fp32(int n, float* input,
                                              float* output) {
  sum_reference(n, input, output);
}

extern "C" __global__ void sum_reference_fp16(int n, half* input,
                                              half* output) {
  sum_reference(n, input, output);
}

template <typename T>
__global__ void sum_backward_reference(int n, T* input, T* input_grad,
                                       T* output_grad) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  input_grad[xid] = output_grad[0];
}

extern "C" __global__ void sum_backward_reference_fp32(int n, float* input,
                                                       float* input_grad,
                                                       float* output_grad) {
  sum_backward_reference(n, input, input_grad, output_grad);
}

extern "C" __global__ void sum_backward_reference_fp16(int n, half* input,
                                                       half* input_grad,
                                                       half* output_grad) {
  sum_backward_reference(n, input, input_grad, output_grad);
}
