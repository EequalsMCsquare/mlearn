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

class SGD(Optimizer):
    """
        随机梯度下降
        Stochastic Gradient Descent
    """
    def __init__(self, module:Module, lr: float = 1e-3) -> None:
        super(SGD,self).__init__(lr, module)
    def step(self) -> None:
        self.iteration += 1
        for parameter in self.module.parameters():
            v = parameter.grad
            parameter -= v * self.lr


class Momentum(Optimizer):
    """
        动量梯度下降
        Momentum
    """
    def __init__(self, module: 'Module', lr: float = 1e-3, momentum: float = 0.9):
        self.module = module
        self.lr = lr
        self.momentum = momentum
        self.v = []
        for parameter in self.module.parameters():
            self.v.append(zeros(*parameter.data.shape))

    def step(self):
        for i, parameter in enumerate(self.module.parameters()):
            v = self.momentum * self.v[i] + (1 - self.momentum) * parameter.grad
            self.v[i] = v
            parameter -= self.lr * v

class RMSProp(Optimizer):
    def __init__(self, module: 'Module', lr: float = 1e-3, beta: float = 0.999, eps: float = 1e-7) -> None:
        """
            Module -> 神经网络模型
            lr -> 学习率
            beta -> 权重衰减速率

             MATH:
                    s = beta * 累计平方梯度 + (1 - beta) * 参数梯度 ** 2
                    eta = self.lr * 参数梯度 / sqrt(s + eps)
                    参数更新 -= eta
        """
        self.module = module
        self.lr = lr
        self.beta = beta
        self.eps = eps # 小常数 1e-7
        self.s = [] # 累计平方梯度
        for parameter in self.module.parameters():
            self.s.append(np.zeros_like(parameter.data))

    def step(self):
        for i,parameter in enumerate(self.module.parameters()):
            s = self.beta * self.s[i] + (1-self.beta) * parameter.grad.data**2
            self.s[i] = s
            eta = self.lr * parameter.grad.data / (np.sqrt(s + self.eps))
            parameter -=   eta

class Adam(Optimizer):
    """
        Adaptive Moment Estimation
    """
    def __init__(self, module: 'Module', lr: float = 1e-3, betas:float=(0.9, 0.999),eps:float=1e-7) -> None:
        super(Adam,self).__init__(lr, module)
        self.betas = betas
        self.eps = eps
        self.v = []
        self.s = []
        for parameter in self.module.parameters():
            self.v.append(np.zeros_like(parameter.data))
            self.s.append(np.zeros_like(parameter.data))


    def step(self) -> None:
        for i,parameter in enumerate(self.module.parameters()):
            v = self.betas[0] * self.v[i] + (1 - self.betas[0]) * parameter.grad.data
            s = self.betas[1] * self.s[i] + (1 - self.betas[1]) * parameter.grad.data ** 2
            self.v[i], self.s[i] = v, s
            eta = self.lr * v/np.sqrt(s+self.eps)
            parameter -= eta
