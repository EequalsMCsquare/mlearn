from ..autograd.tensor import Tensor,Dependency,ensure_tensor


import cupy as cp


"""
################ Mathematic ########################
"""


def exp(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.exp(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def exp_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * data
        depends_on = [Dependency(tensor, exp_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def abs(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.abs(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def abs_backward(grad: cp.ndarray) -> cp.ndarray:
            _self_grad = tensor.data
            _self_grad[_self_grad == 0] = 0
            _self_grad[_self_grad < 0] = -1
            _self_grad[_self_grad > 0] = 1
            return grad * _self_grad
        depends_on = [Dependency(tensor, abs_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def round(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.round(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def round_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * 0
        depends_on = [Dependency(tensor, round_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def ceil(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.ceil(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def ceil_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * 0
        depends_on = [Dependency(tensor, ceil_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def max(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.max(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def max_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * (tensor.data == data).astype(int)
        depends_on = [Dependency(tensor, max_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def mean(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.mean(tensor.data)
    requires_grad = tensor.requires_grad
    _len = tensor.data.flatten().shape[0]
    if requires_grad:
        def mean_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * (1 / _len)
        depends_on = [Dependency(tensor, mean_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def min(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.min(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def min_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * (tensor.data == data).astype(int)
        depends_on = [Dependency(tensor, min_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def sqrt(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.sqrt(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def sqrt_Backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * tensor.data**0.5
        depends_on = [Dependency(tensor,sqrt_Backward)]
    else:
        depends_on = []
    return Tensor(data,requires_grad,depends_on)