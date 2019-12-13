from ..autograd.tensor import Tensor, Dependency, ensure_tensor, ones, zeros
from ..autograd.parameter import Parameter

import cupy as cp
import numpy as np
import warnings
from cupy.lib.stride_tricks import as_strided

"""
################### Layers #######################
"""


def dense(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    assert (inputs.shape[-1] == weights.shape[0]), "傻逼，[形状不匹配]"
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
    data = [((cp.random.uniform(0, 1.0, (mask_shape)) < keep_prob) *
             x).tolist() for x in inputs.data.reshape(-1, mask_shape)]
    requires_grad = inputs.requires_grad

    if requires_grad:
        def grad_fn(grad: cp.ndarray) -> cp.ndarray:
            return (grad * 2) * (cp.array(data) != 0)
        depends_on = [Dependency(inputs, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on).reshape(*inputs.shape)


def flatten(inputs: Tensor):
    batch_size = inputs.shape[0]
    return inputs.reshape(batch_size, -1)


def conv1d(inputs: Tensor, kernel_size: int, stride: int = 1) -> Tensor:
    raise NotImplementedError("???")


def conv_2d(inputs: Tensor, weights: Parameter,
            bias: Parameter = None, stride: int = 1, padding: int = 0) -> Tensor:
    warnings.warn("这cupy版本也太慢了吧， 操!")
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
    out_x = (inputs.shape[2] - weights.shape[2] + 2 * padding) // stride + 1
    out_y = (inputs.shape[3] - weights.shape[3] + 2 * padding) // stride + 1

    if bias is None:
        bias = ones(weights.shape[0], 1, 1)
    else:
        if bias.shape[0] == weights.shape[0] and len(bias.shape) == 1:
            bias = ensure_tensor(bias).reshape(2, 1, 1)
        elif bias.shape == (weights.shape[0], 1, 1):
            bias = ensure_tensor(bias)
        else:
            raise ValueError(
                f"Bias的形状和输出无法匹配, Expect Shape({weights.shape[0]}) 或 Shape{weights.shape[0], 1, 1}. But receive shape({bias.shape})")

    requires_grad = inputs.requires_grad or weights.requires_grad

    view_shape = (1, out_x, out_y,
                  weights.shape[1], weights.shape[2], weights.shape[3])
    batch_result = []
    for sample in inputs.data:
        arr = as_strided(sample, view_shape, sample.strides*2).reshape(view_shape[1:])
        sample_result = []
        for out_dim in weights.data:
            single_channel_conv = []
            for row in arr:
                for col in row:
                    single_channel_conv.append(float((col * out_dim).sum()))
            sample_result.append(
                np.array(single_channel_conv).reshape(out_x, out_y))
        sample_result = cp.array(sample_result) + bias.data
        batch_result.append(cp.asnumpy(sample_result))
    data = cp.array(batch_result)
    if requires_grad:
        raise NotImplementedError("Conv2d 反向传播 Not Implemented yet")

        def conv2d_backward(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on = [Dependency(inputs, conv2d_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
