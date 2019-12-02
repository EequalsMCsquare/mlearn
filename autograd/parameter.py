
import numpy as np


from .tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape) * np.sqrt(1./shape[-1])
        super().__init__(data, requires_grad=True)

    
        