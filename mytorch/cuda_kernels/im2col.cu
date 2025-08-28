#include <cuda_fp16.h>

template <typename T>
__global__ void im2col_input_reference(int batch_size, int input_h, int input_w,
                                       int in_channels, int out_channels,
                                       int kernel_size_h, int kernel_size_w,
                                       int stride_h, int stride_w,
                                       int padding_h, int padding_w, T *input,
                                       T *bias, T *x) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;
  int x_rows = batch_size * output_h * output_w;
  int x_cols =
      in_channels * kernel_size_h * kernel_size_w + int(bias != nullptr);
  int x_cols_padded = (x_cols + 3) / 4 * 4;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;

  if (xid >= x_rows || yid >= x_cols_padded) return;

  int x_row_offset = x_cols_padded * xid;

  for (int i = xid; i < x_rows; i += blockDim.x * gridDim.x)
    for (int j = yid; j < x_cols_padded; j += blockDim.y * gridDim.y) {
      if (j >= x_cols) {
        x[i * x_cols_padded + j] = 0;
      } else if (j == x_cols - 1 && bias != nullptr) {
        int channels_idx = j / (kernel_size_h * kernel_size_w);
        x[i * x_cols_padded + j] = 1;
      } else {
        int output_h_idx = i / output_w % output_h;
        int output_w_idx = i % output_w;
        int batch_idx = i / (output_h * output_w);
        int channels_idx = j / (kernel_size_h * kernel_size_w);
        int kernel_size_h_idx = j / kernel_size_w % kernel_size_h;
        int kernel_size_w_idx = j % kernel_size_w;
        int input_x = output_h_idx * stride_h + kernel_size_h_idx - padding_h;
        int input_y = output_w_idx * stride_w + kernel_size_w_idx - padding_w;
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||
                          input_y >= input_w;
        T input_value =
            is_padding
                ? (T)0
                : input[((batch_idx * in_channels + channels_idx) * input_h +
                         input_x) *
                            input_w +
                        input_y];

        x[i * x_cols_padded + j] = input_value;
      }
    }
}

extern "C" __global__ void im2col_input_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *input, float *bias, float *x) {
  im2col_input_reference(batch_size, input_h, input_w, in_channels,
                         out_channels, kernel_size_h, kernel_size_w, stride_h,
                         stride_w, padding_h, padding_w, input, bias, x);
}

extern "C" __global__ void im2col_input_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *input, half *bias, half *x) {
  im2col_input_reference(batch_size, input_h, input_w, in_channels,
                         out_channels, kernel_size_h, kernel_size_w, stride_h,
                         stride_w, padding_h, padding_w, input, bias, x);
}

template <typename T>
__global__ void im2col_weight_reference(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, T *weight, T *bias, T *y) {
  int y_rows = out_channels;
  int y_cols =
      in_channels * kernel_size_h * kernel_size_w + int(bias != nullptr);
  int y_cols_padded = (y_cols + 3) / 4 * 4;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;

  for (int i = xid; i < y_rows; i += blockDim.x * gridDim.x)
    for (int j = yid; j < y_cols_padded; j += blockDim.y * gridDim.y) {
      if (j >= y_cols) {
        y[i * y_cols_padded + j] = 0;
      } else if (j == y_cols - 1 && bias != nullptr) {
        int out_channels_idx = i;
        y[i * y_cols_padded + j] = bias[out_channels_idx];
      } else {
        int out_channels_idx = i;
        int in_channels_idx = j / (kernel_size_h * kernel_size_w);
        int kernel_size_h_idx = j / kernel_size_w % kernel_size_h;
        int kernel_size_w_idx = j % kernel_size_w;
        T weight_value =
            weight[((out_channels_idx * in_channels + in_channels_idx) *
                        kernel_size_h +
                    kernel_size_h_idx) *
                       kernel_size_w +
                   kernel_size_w_idx];
        y[i * y_cols_padded + j] = weight_value;
      }
    }
}

extern "C" __global__ void im2col_weight_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *weight, float *bias, float *y) {
  im2col_weight_reference(batch_size, input_h, input_w, in_channels,
                          out_channels, kernel_size_h, kernel_size_w, stride_h,
                          stride_w, padding_h, padding_w, weight, bias, y);
}

extern "C" __global__ void im2col_weight_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *weight, half *bias, half *y) {
  im2col_weight_reference(batch_size, input_h, input_w, in_channels,
                          out_channels, kernel_size_h, kernel_size_w, stride_h,
                          stride_w, padding_h, padding_w, weight, bias, y);
}

template <typename T>
__global__ void reverse_im2col_input_reference(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, T *input, T *bias, T *x) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;
  int x_rows = batch_size * output_h * output_w;
  int x_cols =
      in_channels * kernel_size_h * kernel_size_w + int(bias != nullptr);
  int x_cols_padded = (x_cols + 3) / 4 * 4;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;

  if (xid >= x_rows || yid >= x_cols_padded) return;

  int x_row_offset = x_cols_padded * xid;

  for (int i = xid; i < x_rows; i += blockDim.x * gridDim.x)
    for (int j = yid; j < x_cols - int(bias != nullptr);
         j += blockDim.y * gridDim.y) {
      int output_h_idx = i / output_w % output_h;
      int output_w_idx = i % output_w;
      int batch_idx = i / (output_h * output_w);
      int channels_idx = j / (kernel_size_h * kernel_size_w);
      int kernel_size_h_idx = j / kernel_size_w % kernel_size_h;
      int kernel_size_w_idx = j % kernel_size_w;
      int input_x = output_h_idx * stride_h + kernel_size_h_idx - padding_h;
      int input_y = output_w_idx * stride_w + kernel_size_w_idx - padding_w;
      bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||
                        input_y >= input_w;

      if (!is_padding) {
        input[((batch_idx * in_channels + channels_idx) * input_h + input_x) *
                  input_w +
              input_y] += x[i * x_cols_padded + j];
      }
    }
}

extern "C" __global__ void reverse_im2col_input_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *input, float *bias, float *x) {
  reverse_im2col_input_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, input, bias, x);
}

extern "C" __global__ void reverse_im2col_input_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *input, half *bias, half *x) {
  reverse_im2col_input_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, input, bias, x);
}

template <typename T>
__global__ void reverse_im2col_weight_reference(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, T *weight, T *bias, T *y) {
  int y_rows = out_channels;
  int y_cols =
      in_channels * kernel_size_h * kernel_size_w + int(bias != nullptr);
  int y_cols_padded = (y_cols + 3) / 4 * 4;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;

  for (int i = xid; i < y_rows; i += blockDim.x * gridDim.x)
    for (int j = yid; j < y_cols; j += blockDim.y * gridDim.y) {
      if (j == y_cols - 1 && bias != nullptr) {
        int out_channels_idx = i;
        bias[out_channels_idx] = y[i * y_cols_padded + j];
      } else {
        int out_channels_idx = i;
        int in_channels_idx = j / (kernel_size_h * kernel_size_w);
        int kernel_size_h_idx = j / kernel_size_w % kernel_size_h;
        int kernel_size_w_idx = j % kernel_size_w;
        weight[((out_channels_idx * in_channels + in_channels_idx) *
                    kernel_size_h +
                kernel_size_h_idx) *
                   kernel_size_w +
               kernel_size_w_idx] = y[i * y_cols_padded + j];
      }
    }
}

extern "C" __global__ void reverse_im2col_weight_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *weight, float *bias, float *y) {
  reverse_im2col_weight_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, weight, bias, y);
}

extern "C" __global__ void reverse_im2col_weight_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *weight, half *bias, half *y) {
  reverse_im2col_weight_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, weight, bias, y);
}