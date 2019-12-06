import cupy as cp


from .tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = cp.random.randn(*shape) * cp.sqrt(1./shape[-1])
        super().__init__(data, requires_grad=True)
