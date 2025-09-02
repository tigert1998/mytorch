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