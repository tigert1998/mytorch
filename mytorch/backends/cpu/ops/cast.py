from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "cast")
def cpu_cast(x, dtype):
    from mytorch.tensor import Tensor

    output_tensor = Tensor(
        dtype=dtype,
        shape=x.shape,
        device=x.device,
    )
    output_tensor.cpu_array = x._numpy().astype(dtype.np_dtype)
    return output_tensor
