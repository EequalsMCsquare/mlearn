from ..layers import Module
import numpy as np
from ..autograd import Tensor
from ..autograd import tensor

def tensor2array(var):
    assert isinstance(var,Tensor),"必须传入一个Tensor"
    dim = len(var.shape)
    _tmp = []

class Optimizer:
    def __init__(self):
        pass
    

class SGD(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-3) -> None:
        self.lr = lr
        self.module = module

    def step(self) -> None:
        for parameter in self.module.parameters():
            v = parameter.grad
            parameter -= v * self.lr



class RMSProp(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-3, alpha: float = 0.9, eps: float = 1e-8) -> None:
        """
        Module -> 神经网络模型
        lr -> 学习率
        alpha -> 
        """
        self.module = module
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = []
        for parameter in self.module.parameters():
            self.v.append(tensor.zeros_like(parameter.data))

    def step(self):
        v_iter = iter(self.v)
        for parameter in self.module.parameters():
            v = self.alpha * next(v_iter)
            v = v + (1-self.alpha) * parameter.grad**2
            eta = self.lr / (np.sqrt(v.data) + self.eps)
            parameter -=  parameter.grad * eta
           
        

class Momentum(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-3, momentum: float = 0.9):
        self.module = module
        self.lr = lr
        self.momentum = momentum
        self.v = []
        for parameter in self.module.parameters():
            self.v.append(tensor.zeros(*parameter.data.shape))

    def step(self):
        v_iter = iter(self.v)
        for parameter in self.module.parameters():
            v = self.momentum * next(v_iter) + parameter.grad
            parameter -= self.lr * v


class Adam:
    def __init__(self, module: Module, lr: float = 1e-3) -> None:
        raise NotImplementedError("Adam优化器可咋整啊，伙计")
        self.ler = lr
        self.module = module

    def step(self) -> None:
        pass