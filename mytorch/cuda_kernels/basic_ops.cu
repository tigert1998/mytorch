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

template <typename T>
__global__ void sum_reference(int n, T* input, int shape_n, int* shape,
                              int num_reduce_axis, int* reduce_axis,
                              T* output) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  int to = reduce_shape(xid, shape_n, shape, num_reduce_axis, reduce_axis);
  atomicAdd(&output[to], input[xid]);
}

extern "C" __global__ void sum_reference_fp32(int n, float* input, int shape_n,
                                              int* shape, int num_reduce_axis,
                                              int* reduce_axis, float* output) {
  sum_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis, output);
}

extern "C" __global__ void sum_reference_fp16(int n, half* input, int shape_n,
                                              int* shape, int num_reduce_axis,
                                              int* reduce_axis, half* output) {
  sum_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis, output);
}

template <typename T>
__global__ void sum_backward_reference(int n, T* input, int shape_n, int* shape,
                                       int num_reduce_axis, int* reduce_axis,
                                       T* input_grad, T* output_grad) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  int to = reduce_shape(xid, shape_n, shape, num_reduce_axis, reduce_axis);
  input_grad[xid] = output_grad[to];
}

extern "C" __global__ void sum_backward_reference_fp32(
    int n, float* input, int shape_n, int* shape, int num_reduce_axis,
    int* reduce_axis, float* input_grad, float* output_grad) {
  sum_backward_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis,
                         input_grad, output_grad);
}

extern "C" __global__ void sum_backward_reference_fp16(
    int n, half* input, int shape_n, int* shape, int num_reduce_axis,
    int* reduce_axis, half* input_grad, half* output_grad) {
  sum_backward_reference(n, input, shape_n, shape, num_reduce_axis, reduce_axis,
                         input_grad, output_grad);
}

__device__ int permute_shape(int xid, int shape_n, int* shape, int* permute) {
  int to = 0, tmp = xid;
  for (int i = shape_n - 1; i >= 0; i--) {
    int dim = tmp % shape[i];
    tmp /= shape[i];

    int idx;
    for (idx = 0; idx < shape_n; idx++)
      if (permute[idx] == i) break;

    int mul = 1;
    for (int j = shape_n - 1; j > idx; j--) mul = mul * shape[permute[j]];
    to += dim * mul;
  }
  return to;
}

template <typename T>
__global__ void permute_reference(int n, T* input, T* output, int shape_n,
                                  int* shape, int* permute) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int to = permute_shape(xid, shape_n, shape, permute);
  output[to] = input[xid];
}

extern "C" __global__ void permute_reference_fp32(int n, float* input,
                                                  float* output, int shape_n,
                                                  int* shape, int* permute) {
  permute_reference(n, input, output, shape_n, shape, permute);
}

extern "C" __global__ void permute_reference_fp16(int n, half* input,
                                                  half* output, int shape_n,
                                                  int* shape, int* permute) {
  permute_reference(n, input, output, shape_n, shape, permute);
}

template <typename T>
__global__ void permute_backward_reference(int n, int shape_n, int* shape,
                                           int* permute, T* input_grad,
                                           T* output_grad) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int to = permute_shape(xid, shape_n, shape, permute);
  input_grad[xid] = output_grad[to];
}

extern "C" __global__ void permute_backward_reference_fp32(int n, int shape_n,
                                                           int* shape,
                                                           int* permute,
                                                           float* input_grad,
                                                           float* output_grad) {
  permute_backward_reference(n, shape_n, shape, permute, input_grad,
                             output_grad);
}

extern "C" __global__ void permute_backward_reference_fp16(int n, int shape_n,
                                                           int* shape,
                                                           int* permute,
                                                           half* input_grad,
                                                           half* output_grad) {
  permute_backward_reference(n, shape_n, shape, permute, input_grad,
                             output_grad);
}
