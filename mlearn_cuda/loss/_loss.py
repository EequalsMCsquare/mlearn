from ..autograd.tensor import Tensor
import matplotlib.pyplot as plt
from ..functional import loss_func as F
import cupy as cp




class Loss:
    def __init__(self):
        super().__init__()
        self.history = []

    def compute(self,predicts:Tensor, targets:Tensor):
        pass

    def __call__(self,predicts:Tensor, targets:Tensor):
        result = self.compute(predicts,targets)
        self.history.append(cp.asnumpy(result.data))
        return result

    def plot(self):
        plt.show(self.history)
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
        return F.cross_entropy(predicts,targets.long())