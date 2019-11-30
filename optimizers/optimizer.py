from ..layers import Module


class SGD:
    def __init__(self, module: Module, lr: float = 1e-3) -> None:
        self.lr = lr
        self.module = module

    def step(self) -> None:
        for parameter in self.module.parameters():
            parameter -= parameter.grad * self.lr

class Adam:
    def __init__(self, module:Module, lr:float=1e-3) -> None:
        self.ler = lr
        self.module = module

    def step(self) -> None:
        raise NotImplementedError("Adam优化器可咋整啊，伙计")