from ..layers import Module
import numpy as np
from ..autograd import tensor

def tensor2array(var):
    assert isinstance(var,tensor),"必须传入一个Tensor"
    dim = len(var.shape)
    _tmp = []
    

class SGD:
    def __init__(self, module: Module, lr: float = 1e-3) -> None:
        self.lr = lr
        self.module = module

    def step(self) -> None:
        for parameter in self.module.parameters():
            parameter -= parameter.grad * self.lr
            # 之前
            # parameter  = parameter - parameter.grad * self.lr


class RMSProp:
    def __init__(self, module: Module, lr: float = 1e-3, alpha: float = 0.9, eps: float = 1e-8) -> None:
        # raise NotImplementedError("BUG")
        self.module = module
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = None

    def step(self) -> None:
        for parameter in self.module.parameters():
            if self.v is None:
                self.v = np.zeros_like(parameter.data)
            self.v = self.alpha * self.v
            self.v = self.v + (1-self.alpha) * parameter.grad**2
            eta = self.lr / (np.sqrt(self.v) + self.eps)
            parameter -=   parameter.grad * eta
            print(1)


class Momentum:
    def __init__(self, module: Module, lr: float = 1e-3, momentum: float = 0.9):
        raise NotImplementedError("Bug")
        self.module = module
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self):
        for parameter in self.module.parameters():
            if self.v is None:
                self.v = np.zeros_like(parameter.data)

            self.v = self.momentum * self.v + parameter.grad
            parameter -= self.lr * self.v


class Adam:
    def __init__(self, module: Module, lr: float = 1e-3) -> None:
        raise NotImplementedError("Adam优化器可咋整啊，伙计")
        self.ler = lr
        self.module = module

    def step(self) -> None:
        pass