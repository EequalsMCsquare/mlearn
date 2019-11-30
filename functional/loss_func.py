from ..autograd.tensor import Tensor, Dependency, ensure_tensor
from .activation import softmax
import numpy as np
np.set_printoptions(
    suppress=True,
    precision=3,
    formatter={'float': '{:0.4f}'.format}
)


"""
################### Loss Function #########################
"""


def mse(predicts: Tensor, targets: Tensor) -> Tensor:
    assert (predicts.shape ==
            targets.shape), f"[形状不匹配], 预测->{predicts.shape}但是目标却是{targets.shape}"
    tmp = 1
    for x in predicts.shape:
        tmp *= x
    result = ((predicts - targets) ** 2).sum()/tmp
    data = result.data
    requires_grad = result.requires_grad
    if requires_grad:
        depends_on = result.depends_on
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def cross_entropy(predicts: Tensor, targets: Tensor) -> Tensor:
    m = targets.shape[0]
    p = softmax(predicts).data
    log_likelihood = -np.log(p[range(m), targets.data])
    data = np.sum(log_likelihood) / m

    requires_grad = predicts.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            p = softmax(predicts).data
            p[range(m), targets.data] -= 1
            p = p/m
            return p
        depends_on = [Dependency(predicts, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)
