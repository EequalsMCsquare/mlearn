from ..autograd.tensor import Tensor, Dependency, ensure_tensor
from ..autograd.parameter import Parameter

import numpy as np
np.set_printoptions(
    suppress=True,
    precision=3,
    formatter={'float': '{:0.4f}'.format}
)

"""
################### Layers #######################
"""


def dense(inputs: Tensor, weights: Tensor, bias: Tensor):
    assert (inputs.shape[-1] == weights.shape[0]), "傻逼，[形状不匹配]"
    return inputs @ weights + bias


# TODO: Dropout 没有考虑对于多维的处理
"""
思路： 先将其reshape成(-1, shape[-1]) 再进行mask最后还原回去
"""


def dropout_1d(inputs: Tensor, keep_prob: float) -> Tensor:
    assert (inputs.data.ndim > 1), "没有足够的维度来进行1D Dropout"
    mask_shape = inputs.shape[-1]
    data = [(np.random.uniform(0, 1.0, (mask_shape)) < keep_prob) *
            x for x in inputs.data.reshape(-1, mask_shape)]
    requires_grad = inputs.requires_grad

    if requires_grad:
        def grad_fn(grad:np.ndarray) -> np.ndarray:
            return (grad * 2) * (np.array(data)!= 0)
        depends_on = [Dependency(inputs,grad_fn)]

    return Tensor(data, requires_grad, depends_on).reshape(*inputs.shape)


def dropout_2d(inputs: Tensor, keep_prob: float) -> Tensor:
    raise NotImplementedError("还没弄好呢 弟弟")
    assert (inputs.data.ndim > 2), "没有足够的维度来进行2D Dropout"
    mask_shape = inputs.shape[-2:]

    return (inputs * (np.random.uniform(0, 1.0, mask_shape) < keep_prob))
