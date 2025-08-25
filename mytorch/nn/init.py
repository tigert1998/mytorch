def _calculate_fan_in_and_fan_out(tensor):
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    for dim in tensor.shape[2:]:
        fan_in *= dim
        fan_out *= dim
    return fan_in, fan_out


def kaiming_uniform_(tensor, a=0, mode="fan_in"):
    if mode == "fan_in":
        fan, _ = _calculate_fan_in_and_fan_out(tensor)
    elif mode == "fan_out":
        _, fan = _calculate_fan_in_and_fan_out(tensor)
    bound = (6 / (1 + a**2) / fan) ** 0.5

    tensor.uniform_(-bound, bound)
