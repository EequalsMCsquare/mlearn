from typing import List

import numpy as np

from ..autograd.parameter import Parameter
from ..autograd.tensor import Dependency, Tensor, ensure_tensor, ones, zeros
from .c_func import c_func


"""
################### Layers #######################
"""


def dense(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    for x in(inputs, weights, bias):
        if not isinstance(x, Tensor):
            raise TypeError("传播对象必须是一个Tensor!")

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
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return (grad * 2) * (np.array(data) != 0)
        depends_on = [Dependency(inputs, grad_fn)]

    return Tensor(data, requires_grad, depends_on).reshape(*inputs.shape)


def dropout_2d(inputs: Tensor, keep_prob: float) -> Tensor:
    raise NotImplementedError("还没弄好呢 弟弟")
    assert (inputs.data.ndim > 2), "没有足够的维度来进行2D Dropout"
    mask_shape = inputs.shape[-2:]

    return (inputs * (np.random.uniform(0, 1.0, mask_shape) < keep_prob))


def flatten(inputs: Tensor):
    batch_size = inputs.shape[0]
    return inputs.reshape(batch_size, -1)



def conv_2d(inputs: Tensor, weights: Parameter,
              bias: Parameter = None, stride: int = 1, padding: int = 0) -> Tensor:
    # 内存错位的原因在这里！！！
    # 因为C只能全部接受同样维度的数组
    # 所以我reshape了bias 和weights的， 但是忘记修改index了。。。操！
    # 奇怪 结果有可能是正确的但是， 有一定几率前4位的指针有问题

    """
    inputs -> (Batch Size, Channel, Rows, Cols)
    weights -> (out_channel, in_channel, Rows, Cols)
    bias -> (out_channel, out_x, out_y)
    公式计算卷积后输出的形状
    height_out = (height_in - height_w + 2 * padding) // stride + 1
    width_out = (width_in - width_w + 2 * padding) // stride + 1
    out -> (Batch_size, out_channel, height_out, width_out)
    """
    inputs = ensure_tensor(inputs)
    weights = ensure_tensor(weights)
    if weights.shape[1] != inputs.shape[1]:
        raise ValueError(f"Weights的in_channel和inputs的不匹配, inputs -> {inputs.shape}, weights -> {weights.shape}")

    if bias is None:
        bias = ones(weights.shape[1], 1, 1)
    else:
        if bias.shape[0] == weights.shape[0] and bias.ndim == 1:
            bias = ensure_tensor(bias)
        else:
            raise ValueError(
                f"Bias的形状和输出无法匹配, Expect Shape({weights.shape[0]}) 或 Shape{weights.shape[0], 1, 1}. But receive shape({bias.shape})")

    out_x = (inputs.shape[2] - weights.shape[2] + 2 * padding) // stride + 1
    out_y = (inputs.shape[3] - weights.shape[3] + 2 * padding) // stride + 1

    # TODO Padding
    weights = weights.reshape(1, *weights.shape)
    bias = bias.reshape(-1, 1, 1, 1, 1)
    requires_grad = inputs.requires_grad or weights.requires_grad or bias.requires_grad

    view_shape = (1, out_x, out_y,
                  weights.shape[2], weights.shape[3], weights.shape[4])
    batch_result = []
    # TODO MultiProcess
    for sample in inputs.data:
        arr = np.lib.stride_tricks.as_strided(
            sample, view_shape, sample.strides*2).reshape(view_shape[1:])
        batch_result.append(c_func.sample_conv2d(arr, weights.data, bias.data))

    data = np.array(batch_result)
    depends_on: List[Dependency] = []

    """
    二维卷积的反向传播
    """
   # inputs
    if inputs.requires_grad:
        def conv2d_inputsBackward(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on.append(Dependency(inputs, conv2d_inputsBackward))

    # weights:
    if weights.requires_grad:
        def conv2d_weightsBackward(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on.append(Dependency(weights, conv2d_weightsBackward))
    # bias:
    if bias.requires_grad:
        def conv2d_biasBackward(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on.append(Dependency(bias, conv2d_biasBackward))

    return Tensor(data, requires_grad, depends_on)
