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

#define BROADCAST_BINARY_OPERATION_FORWARD(name, arg_type, arg, op)          \
  template <typename T>                                                      \
  __global__ void name##_reference(int n, int x_shape_n, int* x_shape,       \
                                   int y_shape_n, int* y_shape, T* x, T* y,  \
                                   arg_type(T) T* output) {                  \
    int xid = blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (xid >= n) return;                                                    \
    int2 pair = broadcast(xid, x_shape_n, x_shape, y_shape_n, y_shape);      \
    output[xid] = op();                                                      \
  }                                                                          \
  extern "C" __global__ void name##_reference_fp32(                          \
      int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape,       \
      float* x, float* y, arg_type(float) float* output) {                   \
    name##_reference<float>(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y, \
                            arg() output);                                   \
  }                                                                          \
  extern "C" __global__ void name##_reference_fp16(                          \
      int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape,       \
      half* x, half* y, arg_type(half) half* output) {                       \
    name##_reference<half>(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y,  \
                           arg() output);                                    \
  }

#define BROADCAST_BINARY_OPERATION_BACKWARD(name, arg_type, arg, op_backward)  \
  template <typename T>                                                        \
  __global__ void name##_backward_reference(                                   \
      int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape, T* x,   \
      T* y, arg_type(T) T* x_grad, T* y_grad, T* output_grad) {                \
    int xid = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (xid >= n) return;                                                      \
    int2 pair = broadcast(xid, x_shape_n, x_shape, y_shape_n, y_shape);        \
    op_backward();                                                             \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp32(                   \
      int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape,         \
      float* x, float* y, arg_type(float) float* x_grad, float* y_grad,        \
      float* output_grad) {                                                    \
    name##_backward_reference(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y, \
                              arg() x_grad, y_grad, output_grad);              \
  }                                                                            \
  extern "C" __global__ void name##_backward_reference_fp16(                   \
      int n, int x_shape_n, int* x_shape, int y_shape_n, int* y_shape,         \
      half* x, half* y, arg_type(half) half* x_grad, half* y_grad,             \
      half* output_grad) {                                                     \
    name##_backward_reference(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y, \
                              arg() x_grad, y_grad, output_grad);              \
  }

#define BROADCAST_BINARY_OPERATION(name, arg_type, arg, op, op_backward) \
  BROADCAST_BINARY_OPERATION_FORWARD(name, arg_type, arg, op)            \
  BROADCAST_BINARY_OPERATION_BACKWARD(name, arg_type, arg, op_backward)

#define ALPHA_ARG_TYPE(T) T alpha,
#define ALPHA_ARG() alpha,
#define NO_ARG_TYPE(T)
#define NO_ARG()

#define ADD_WITH_ALPHA() (x[pair.x] + y[pair.y] * alpha)
#define ADD_WITH_ALPHA_BACKWARD()                         \
  do {                                                    \
    atomicAdd(&x_grad[pair.x], output_grad[xid]);         \
    atomicAdd(&y_grad[pair.y], output_grad[xid] * alpha); \
  } while (0)
BROADCAST_BINARY_OPERATION(add, ALPHA_ARG_TYPE, ALPHA_ARG, ADD_WITH_ALPHA,
                           ADD_WITH_ALPHA_BACKWARD)

#define SUB_WITH_ALPHA() (x[pair.x] - y[pair.y] * alpha)
#define SUB_WITH_ALPHA_BACKWARD()                          \
  do {                                                     \
    atomicAdd(&x_grad[pair.x], output_grad[xid]);          \
    atomicAdd(&y_grad[pair.y], -output_grad[xid] * alpha); \
  } while (0)
BROADCAST_BINARY_OPERATION(sub, ALPHA_ARG_TYPE, ALPHA_ARG, SUB_WITH_ALPHA,
                           SUB_WITH_ALPHA_BACKWARD)

#define MUL() (x[pair.x] * y[pair.y])
#define MUL_BACKWARD()                                        \
  do {                                                        \
    atomicAdd(&x_grad[pair.x], output_grad[xid] * y[pair.y]); \
    atomicAdd(&y_grad[pair.y], output_grad[xid] * x[pair.x]); \
  } while (0)
BROADCAST_BINARY_OPERATION(mul, NO_ARG_TYPE, NO_ARG, MUL, MUL_BACKWARD)

#define DIV() (x[pair.x] / y[pair.y])
#define DIV_BACKWARD()                                                 \
  do {                                                                 \
    atomicAdd(&x_grad[pair.x], output_grad[xid] / y[pair.y]);          \
    atomicAdd(&y_grad[pair.y],                                         \
              output_grad[xid] * (-x[pair.x] / (T)pow(y[pair.y], 2))); \
  } while (0)
BROADCAST_BINARY_OPERATION(div, NO_ARG_TYPE, NO_ARG, DIV, DIV_BACKWARD)

#define POW() ((T)pow((float)x[pair.x], (float)y[pair.y]))
#define POW_BACKWARD()                                                        \
  do {                                                                        \
    atomicAdd(&x_grad[pair.x],                                                \
              output_grad[xid] * y[pair.y] *                                  \
                  (T)pow((float)x[pair.x], (float)y[pair.y] - 1));            \
    atomicAdd(&y_grad[pair.y],                                                \
              output_grad[xid] * (T)pow((float)x[pair.x], (float)y[pair.y]) * \
                  (T)logf((float)x[pair.x]));                                 \
  } while (0)
BROADCAST_BINARY_OPERATION(pow, NO_ARG_TYPE, NO_ARG, POW, POW_BACKWARD)

#define COPY() y[pair.y]
BROADCAST_BINARY_OPERATION_FORWARD(copy, NO_ARG_TYPE, NO_ARG, COPY)
