from ..autograd.tensor import Tensor, Dependency, ensure_tensor

import numpy as np


"""
###################### Activation Functions ###################
"""


def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data**2)

        depends_on = [Dependency(tensor, grad_fn)]

    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def relu(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = np.where(tensor.data > 0, tensor.data, 0)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.where(data <= 0, 0, 1)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def leaky_relu(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)

    data = np.where(tensor.data > 0, tensor.data, 0.01 * tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.where(data <= 0, 0.01, 1)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def sigmoid(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    data = _sigmoid(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * _sigmoid(data) * (1 - _sigmoid(data))
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def softmax(tensor: Tensor, dim: int = 1) -> Tensor:
    tensor = ensure_tensor(tensor)

    def _stable_softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        _sum = np.sum(np.exp(x))
        return np.exp(x) / _sum
    data = []

    if dim == 1:
        _tmp_data = tensor.data
    elif dim == 0:
        _tmp_data = tensor.data.T
    else:
        raise RuntimeError("请输入有效的Dim!!!")
    for x in _tmp_data:
        data.append(_stable_softmax(x))
    data = np.array(data) if dim == 1 else np.array(data).T
    requires_grad = tensor.requires_grad

    if tensor.requires_grad:
        raise NotImplementedError("梯度！！！")

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, tensor.depends_on)
