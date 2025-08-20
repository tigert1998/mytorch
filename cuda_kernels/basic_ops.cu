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

template <typename T>
__global__ void fill_reference(int n, T* input, T value) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  input[xid] = value;
}

extern "C" __global__ void fill_reference_fp32(int n, float* input,
                                               float value) {
  fill_reference(n, input, value);
}

extern "C" __global__ void fill_reference_fp16(int n, half* input, half value) {
  fill_reference(n, input, value);
}

template <typename T>
__global__ void permute_reference(int n, T* input, T* output, int shape_n,
                                  int* permute, int* shape) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int to = 0, tmp = xid;
  for (int i = shape_n - 1; i >= 0; i--) {
    int dim = tmp % shape[i];
    tmp /= shape[i];

    int idx;
    for (idx = 0; idx < shape_n; idx++)
      if (permute[idx] == i) break;
    // permute[idx] = i
    // i -> idx

    int mul = 1;
    for (int j = shape_n - 1; j > idx; j--) mul = mul * shape[permute[j]];
    to += dim * mul;
  }

  output[to] = input[xid];
}

extern "C" __global__ void permute_reference_fp32(int n, float* input,
                                                  float* output, int shape_n,
                                                  int* permute, int* shape) {
  permute_reference(n, input, output, shape_n, permute, shape);
}

extern "C" __global__ void permute_reference_fp16(int n, half* input,
                                                  half* output, int shape_n,
                                                  int* permute, int* shape) {
  permute_reference(n, input, output, shape_n, permute, shape);
}

template <typename T>
__global__ void permute_backward_reference(int n, int shape_n, int* permute,
                                           int* shape, T* input_grad,
                                           T* output_grad) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int to = 0, tmp = xid;
  for (int i = shape_n - 1; i >= 0; i--) {
    int dim = tmp % shape[i];
    tmp /= shape[i];

    int idx;
    for (idx = 0; idx < shape_n; idx++)
      if (permute[idx] == i) break;
    // permute[idx] = i
    // i -> idx

    int mul = 1;
    for (int j = shape_n - 1; j > idx; j--) mul = mul * shape[permute[j]];
    to += dim * mul;
  }

  input_grad[xid] = output_grad[to];
}

extern "C" __global__ void permute_backward_reference_fp32(int n, int shape_n,
                                                           int* permute,
                                                           int* shape,
                                                           float* input_grad,
                                                           float* output_grad) {
  permute_backward_reference(n, shape_n, permute, shape, input_grad,
                             output_grad);
}

extern "C" __global__ void permute_backward_reference_fp16(int n, int shape_n,
                                                           int* permute,
                                                           int* shape,
                                                           half* input_grad,
                                                           half* output_grad) {
  permute_backward_reference(n, shape_n, permute, shape, input_grad,
                             output_grad);
}
