from ..autograd.tensor import Tensor
import matplotlib.pyplot as plt
import numpy as np
from ..functional import loss_func as F




class Loss:
    def __init__(self):
        super().__init__()
        self.hist = []

    def compute(self,predicts:Tensor, targets:Tensor):
        pass

    def __call__(self,predicts:Tensor, targets:Tensor):
        result = self.compute(predicts,targets)
        self.hist.append(result.data)
        return result

    def plot(self):
        plt.show(self.hist)
        plt.show()

class MSE(Loss):
    def __init__(self):
        super().__init__()
    
    def compute(self,predicts:Tensor,targets:Tensor):
        return F.mse(predicts,targets)

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def compute(self,predicts:Tensor,targets:Tensor):
        return F.cross_entropy(predicts,targets)
