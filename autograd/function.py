from autograd.tensor import Tensor, Dependency

import numpy as np
np.set_printoptions(
    suppress = True,
    precision = 3,
    formatter = {'float':'{:0.4f}'.format}
    )

"""
################ Mathematic ########################
exp
abs
round
ceil

sin
cos
tan

#################### Activation ####################
tanh [Y] -> Line 30
ReLU [Y] -> Line 45
leaky ReLU [Y] -> Line 57
SELU 
sigmoid [Y] -> Line 69
softmax [50%] -> Line 84

################### Loss Func #######################
MSE [N]
Cross Entropy
Categorical Cross Entropy
Binary Cross Entropy

################## Layers ##########################
Dense 
Conv1d
Conv2d
MaxPool1d
MaxPool2d

"""

def exp(tensor: Tensor) -> Tensor:
    data = np.exp(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data
        depends_on = [Dependency(tensor,grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

"""
###################### Activation Functions ###################
"""

def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(tensor, grad_fn)]

    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def relu(tensor: Tensor) -> Tensor:
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
    data = np.where(tensor.data > 0, tensor.data, 0.01 * tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.where(data <= 0, 0.01, 1)
        depends_on = [Dependency(tensor,grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def sigmoid(tensor: Tensor) -> Tensor:
    def _sigmoid(x:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    data = _sigmoid(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return grad * _sigmoid(data) * (1 - _sigmoid(data))
        depends_on = [Dependency(tensor,grad_fn)]        
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def softmax(tensor: Tensor, dim:int) -> Tensor:
    def _stable_softmax(x:np.ndarray) -> np.ndarray:
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
    data = np.array(data) if dim==1 else np.array(data).T
    requires_grad = tensor.requires_grad
    depends_on = []
    # TODO: 梯度
    return Tensor(data, requires_grad, depends_on)
