inline __device__ int restore_reduction(int shape_n, int* shape,
                                        int num_reduce_axis, int* reduce_axis,
                                        int outer_idx, int inner_idx) {
  int mul = 1, dest = 0;

  for (int i = shape_n - 1, j = num_reduce_axis - 1; i >= 0; i--) {
    if (j >= 0 && i == reduce_axis[j]) {
      dest += (inner_idx % shape[i]) * mul;
      inner_idx /= shape[i];
      j--;
    } else {
      dest += (outer_idx % shape[i]) * mul;
      outer_idx /= shape[i];
    }
    mul *= shape[i];
  }

  return dest;
}
