from ..layers import Module
import numpy as np
from ..autograd import Tensor, zeros, zeros_like
from ..autograd import Parameter
from typing import Generator


def tensor2array(var):
    assert isinstance(var,Tensor),"必须传入一个Tensor"
    dim = len(var.shape)
    _tmp = []

class Optimizer:
    def __init__(self, lr:float, module:Module):
        self.lr = lr
        self.module = module
        self.iteration = 0

    def step(self):
        """
            更新參數
            optimizer.step()
        """
        raise NotImplementedError("Overrive this method before calling!")

class BGD(Optimizer):
    """
        批梯度下降法
        Batch Gradient Descent
    """
    def __init__(self, module:Module, lr: float = 1e-3) -> None:
        super(BGD,self).__init__(lr, module)
    def step(self) -> None:
        self.iteration += 1
        for parameter in self.module.parameters():
            v = parameter.grad
            parameter -= v * self.lr

class SGD(Optimizer):
    """
        随机梯度下降
        Stochastic Gradient Descent
    """
    def __init__(self, lr, module):
        super(SGD,self).__init__(lr, module)
        raise NotImplementedError("Not Implement!")


class Momentum(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-3, momentum: float = 0.9):
        self.module = module
        self.lr = lr
        self.momentum = momentum
        self.v = []
        for parameter in self.module.parameters():
            self.v.append(zeros(*parameter.data.shape))

    def step(self):
        v_iter = iter(self.v)
        for i, parameter in enumerate(self.module.parameters()):
            v = self.momentum * next(v_iter) + (1 - self.momentum) * parameter.grad
            self.v[i] = v
            parameter -= self.lr * v

class RMSProp(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-3, alpha: float = 0.9, eps: float = 1e-7) -> None:
        """
        Module -> 神经网络模型
        lr -> 学习率
        alpha -> 衰减速率

         MATH:
                v = alpha * 累计平方梯度 + (1 - alpha) * 参数梯度 ** 2
                self.v[i] = (lr/sqrt(eps + v)) * 参数梯度
                参数更新 -= self.v[i]
        """
        self.module = module
        self.lr = lr
        self.alpha = alpha
        self.eps = eps # 小常数 1e-7
        self.v = [] # 累计平方梯度
        for parameter in self.module.parameters():
            self.v.append(zeros_like(parameter.data))

    def step(self):
        for i,parameter in enumerate(self.module.parameters()):
            v = self.alpha * self.v[i]
            v = v + (1-self.alpha) * parameter.grad**2
            self.v[i] = v
            eta = self.lr / (np.sqrt(v.data) + self.eps)
            parameter -=  parameter.grad * eta


class Adam:
    def __init__(self, module: Module, lr: float = 1e-3) -> None:
        raise NotImplementedError("Adam优化器可咋整啊，伙计")
        self.lr = lr
        self.module = module

    def step(self) -> None:
        pass
