from ..autograd.tensor import Tensor, Dependency, ensure_tensor
from ..autograd.parameter import Parameter

import cupy as cp

"""
################### Layers #######################
"""


def dense(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    assert (inputs.shape[-1] == weights.shape[0]), "傻逼，[形状不匹配]"
    for x in(inputs,weights,bias):
        if not isinstance(x,Tensor):
            raise TypeError("传播对象必须是一个Tensor!")

    return inputs @ weights + bias

# TODO: Dropout 没有考虑对于多维的处理
"""
思路： 先将其reshape成(-1, shape[-1]) 再进行mask最后还原回去
"""


def dropout_1d(inputs: Tensor, keep_prob: float) -> Tensor:
    assert (inputs.data.ndim > 1), "没有足够的维度来进行1D Dropout"
    mask_shape = inputs.shape[-1]
    data = [((cp.random.uniform(0, 1.0, (mask_shape)) < keep_prob) *
            x).tolist() for x in inputs.data.reshape(-1, mask_shape)]
    requires_grad = inputs.requires_grad

    if requires_grad:
        def grad_fn(grad:cp.ndarray) -> cp.ndarray:
            return (grad * 2) * (cp.array(data)!= 0)
        depends_on = [Dependency(inputs,grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on).reshape(*inputs.shape)


def flatten(inputs:Tensor):
    batch_size = inputs.shape[0]
    return inputs.reshape(batch_size,-1)
