from ..autograd.tensor import Tensor, Dependency, ensure_tensor
from .activation import softmax as _softmax
import numpy as np




"""
################### Loss Function #########################
"""


def mse(predicts: Tensor, targets: Tensor) -> Tensor:
    assert (predicts.shape ==
            targets.shape), f"[形状不匹配], 预测->{predicts.shape}但是目标却是{targets.shape}"
    tmp = 1
    for x in predicts.shape:
        tmp *= x
    # result = [data**2 for data in (predicts - targets)]#.sum()/tmp
    result = ((predicts - targets)**2) .sum()/tmp
    # return result

    data = result.data
    requires_grad = result.requires_grad
    if requires_grad:
        depends_on = result.depends_on
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)



def cross_entropy(predicts: Tensor, targets: Tensor) -> Tensor:
    def softmax(tensor: Tensor):
        # Stable softmax without grad compute
        def _stable_softmax(x: np.ndarray) -> np.ndarray:
            x = x - np.max(x)
            _sum = np.sum(np.exp(x))
            return np.exp(x) / _sum
        data = []
        for x in tensor:
            data.append(_stable_softmax(x.data))
        data = np.array(data)
        return Tensor(data, tensor.requires_grad,
                      [Dependency(tensor, lambda x: x)])

    m = targets.shape[0]
    p = softmax(predicts).data
    p[p == 0] = 1e-8 * np.random.randint(1, 10)
    log_likelihood = -np.log(p[range(m), targets.data])
    data = np.sum(log_likelihood) / m

    requires_grad = predicts.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            p = softmax(predicts).data
            p[range(m), targets.data] -= 1
            p = p/m
            return p * grad
        depends_on = [Dependency(predicts, grad_fn)]
    else:
        depends_on = []
    return (Tensor(data, requires_grad, depends_on))
