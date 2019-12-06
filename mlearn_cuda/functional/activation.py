from ..autograd.tensor import Tensor, Dependency, ensure_tensor

import cupy as cp 

"""
###################### Activation Functions ###################
"""


def tanh(tensor: Tensor) -> Tensor:
    assert isinstance(tensor, Tensor), "只能接受Tensor对象"
    data = cp.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def tanh_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * (1 - data**2)

        depends_on = [Dependency(tensor, tanh_backward)]

    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def relu(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)
    data = cp.where(tensor.data > 0, tensor.data, 0)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def ReLU_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * cp.where(data <= 0, 0, 1)
        depends_on = [Dependency(tensor, ReLU_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def leaky_relu(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)

    data = cp.where(tensor.data > 0, tensor.data, 0.01 * tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def LReLU_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * cp.where(data <= 0, 0.01, 1)
        depends_on = [Dependency(tensor, LReLU_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def sigmoid(tensor: Tensor) -> Tensor:
    tensor = ensure_tensor(tensor)

    def _sigmoid(x: cp.ndarray) -> cp.ndarray:
        return 1 / (1 + cp.exp(-x))

    data = _sigmoid(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def sigmoid_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * _sigmoid(data) * (1 - _sigmoid(data))
        depends_on = [Dependency(tensor, sigmoid_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)



def softmax(tensor: Tensor, dim: int = 1) -> Tensor:
    tensor = ensure_tensor(tensor)

    def _stable_softmax(x: cp.ndarray) -> cp.ndarray:
        x = x - cp.max(x)
        _sum = cp.sum(cp.exp(x))
        return cp.exp(x) / _sum
    data = []

    if dim == 1:
        _tmp_data = tensor.data
    elif dim == 0:
        _tmp_data = tensor.data.T
    else:
        raise RuntimeError("请输入有效的Dim!!!")
    for x in _tmp_data:
        data.append(_stable_softmax(x).tolist())
    data = cp.array(data) if dim == 1 else cp.array(data).T
    requires_grad = tensor.requires_grad

    if tensor.requires_grad:
        raise NotImplementedError("梯度！！！")
        def softmax_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad
        depends_on = [Dependency(tensor, softmax_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, tensor.depends_on)