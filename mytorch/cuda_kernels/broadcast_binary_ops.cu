#include <cuda_fp16.h>

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
