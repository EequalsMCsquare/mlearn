from ..autograd.tensor import Tensor, Dependency, ensure_tensor

import numpy as np
np.set_printoptions(
    suppress=True,
    precision=3,
    formatter={'float': '{:0.4f}'.format}
)

"""
################ Mathematic ########################
"""


def exp(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.exp(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def abs(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.abs(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            _self_grad = tensor.data
            _self_grad[_self_grad == 0] = 0
            _self_grad[_self_grad < 0] = -1
            _self_grad[_self_grad > 0] = 1
            return grad * _self_grad
            print(_self_grad)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def round(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.round(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * 0
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def ceil(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.ceil(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * 0
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def max(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.max(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (tensor.data == data).astype(int)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def mean(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.mean(tensor.data)
    requires_grad = tensor.requires_grad
    _len = tensor.data.flatten().shape[0]
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 / _len)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def min(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.min(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (tensor.data == data).astype(int)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

