from typing import Tuple

import numpy as np


def calculate_broadcast_shape(
    x_shape: tuple[int, ...], y_shape: tuple[int, ...]
) -> Tuple[int, ...]:
    error_msg = f"Invalid broadcast shape: {x_shape} and {y_shape}"
    if len(x_shape) < len(y_shape):
        x_shape = (1,) * (len(y_shape) - len(x_shape)) + x_shape
    elif len(x_shape) > len(y_shape):
        y_shape = (1,) * (len(x_shape) - len(y_shape)) + y_shape
    ans = []
    for i, j in zip(x_shape, y_shape):
        if not (i == j or i == 1 or j == 1):
            raise RuntimeError(error_msg)
        ans.append(max(i, j))
    return tuple(ans)


def calculate_reduce_shape(shape, dim, keepdim):
    if not np.all([0 <= i and i < len(shape) and isinstance(i, int) for i in dim]):
        raise RuntimeError(f"Reduce dim is invalid: {dim}")
    if keepdim:
        shape = [(1 if i in dim else shape_i) for i, shape_i in enumerate(shape)]
    else:
        shape = [shape_i for i, shape_i in enumerate(shape) if i not in dim]
    return tuple(shape)


def calculate_reshaped_shape(
    original_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
):
    from mytorch.tensor import shape_size

    total_elements = shape_size(original_shape)
    target_elements = 1
    unknown_dim_index = None

    for i, dim in enumerate(target_shape):
        if dim == -1:
            if unknown_dim_index is not None:
                raise ValueError("can only specify one unknown dimension")
            unknown_dim_index = i
        else:
            if dim <= 0:
                raise ValueError("negative dimensions not allowed except -1")
            target_elements *= dim

    if unknown_dim_index is not None:
        if total_elements % target_elements != 0:
            raise ValueError(
                f"cannot reshape array of size {total_elements} into shape {target_shape}"
            )
        unknown_dim = total_elements // target_elements
        target_shape_ls = list(target_shape)
        target_shape_ls[unknown_dim_index] = unknown_dim
        return tuple(target_shape_ls)
    else:
        if total_elements != target_elements:
            raise ValueError(
                f"cannot reshape array of size {total_elements} into shape {target_shape}"
            )
        return target_shape


def calculate_cat_shape(tensors, dim):
    requires_grad = tensors[0].requires_grad
    dtype = tensors[0].dtype
    device = tensors[0].device
    shape = list(tensors[0].shape)
    for i in range(1, len(tensors)):
        error_msg = f"Tensor #{i} mismatches from tensor #0 in cat operator"
        if not (
            requires_grad == tensors[i].requires_grad
            and dtype == tensors[i].dtype
            and device == tensors[i].device
            and len(shape) == len(tensors[i].shape)
        ):
            raise RuntimeError(error_msg)
        for j in range(len(shape)):
            if j != dim and shape[j] != tensors[i].shape[j]:
                raise RuntimeError(error_msg)
        shape[dim] += tensors[i].shape[dim]

    return shape
