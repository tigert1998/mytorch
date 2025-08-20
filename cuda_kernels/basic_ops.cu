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

__device__ int2 broadcast(int idx, int x_shape_n, int* x_shape, int y_shape_n,
                          int* y_shape) {
  int tmp = idx;
  int x_mul = 1;
  int y_mul = 1;
  int x = 0;
  int y = 0;
  for (int i = x_shape_n - 1, j = y_shape_n - 1; i >= 0 || j >= 0; i--, j--) {
    if (i >= 0 && j >= 0) {
      int shape = max(x_shape[i], y_shape[j]);
      int dim = tmp % shape;
      tmp /= shape;
      int x_dim = min(dim, x_shape[i] - 1);
      int y_dim = min(dim, y_shape[j] - 1);
      x += x_dim * x_mul;
      y += y_dim * y_mul;
      x_mul *= x_shape[i];
      y_mul *= y_shape[j];
    } else if (i >= 0) {
      int shape = x_shape[i];
      int dim = tmp % shape;
      tmp /= shape;
      x += dim * x_mul;
      x_mul *= shape;
    } else if (j >= 0) {
      int shape = y_shape[j];
      int dim = tmp % shape;
      tmp /= shape;
      y += dim * y_mul;
      y_mul *= shape;
    }
  }
  return {x, y};
}

template <typename T>
__global__ void add_reference(int n, int x_shape_n, int* x_shape, int y_shape_n,
                              int* y_shape, T* x, T* y, T alpha, T* output) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int2 pair = broadcast(xid, x_shape_n, x_shape, y_shape_n, y_shape);

  output[xid] = x[pair.x] + y[pair.y] * alpha;
}

extern "C" __global__ void add_reference_fp32(int n, int x_shape_n,
                                              int* x_shape, int y_shape_n,
                                              int* y_shape, float* x, float* y,
                                              float alpha, float* output) {
  add_reference<float>(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y, alpha,
                       output);
}

extern "C" __global__ void add_reference_fp16(int n, int x_shape_n,
                                              int* x_shape, int y_shape_n,
                                              int* y_shape, half* x, half* y,
                                              half alpha, half* output) {
  add_reference<half>(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y, alpha,
                      output);
}

template <typename T>
__global__ void add_backward_reference(int n, int x_shape_n, int* x_shape,
                                       int y_shape_n, int* y_shape, T alpha,
                                       T* x_grad, T* y_grad, T* output_grad) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int2 pair = broadcast(xid, x_shape_n, x_shape, y_shape_n, y_shape);

  atomicAdd(&x_grad[pair.x], output_grad[xid]);
  atomicAdd(&y_grad[pair.y], output_grad[xid] * alpha);
}

extern "C" __global__ void add_backward_reference_fp32(
    int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape,
    float alpha, float* x_grad, float* y_grad, float* output_grad) {
  add_backward_reference(n, x_shape_n, x_shape, y_shape_n, y_shape, alpha,
                         x_grad, y_grad, output_grad);
}

extern "C" __global__ void add_backward_reference_fp16(
    int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape, half alpha,
    half* x_grad, half* y_grad, half* output_grad) {
  add_backward_reference(n, x_shape_n, x_shape, y_shape_n, y_shape, alpha,
                         x_grad, y_grad, output_grad);
}
