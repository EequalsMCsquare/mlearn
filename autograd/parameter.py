
import numpy as np


from .tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.rand(*shape)
        super().__init__(data, requires_grad=True)

    
        