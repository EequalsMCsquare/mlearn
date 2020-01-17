from typing import List

import numpy as np
from time import time
from numba import jit, njit, prange

from ..autograd.parameter import Parameter
from ..autograd.tensor import Dependency, Tensor, ensure_tensor, ones
from .c_func import c_func


"""
################### Layers #######################
"""


def dense(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    for x in(inputs, weights, bias):
        if not isinstance(x, Tensor):
            raise TypeError(f"传播对象必须是一个Tensor! 不支持 {type(x)}")

    return (inputs @ weights + bias).no_leaf()


def dense_experiment(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    for x in(inputs, weights, bias):
        if not isinstance(x, Tensor):
            raise TypeError(f"传播对象必须是一个Tensor! 不支持 {type(x)}")

    data = inputs.data @ weights.data + bias.data

    requires_grad = inputs.requires_grad or weights.requires_grad or bias.requires_grad
    depends_on = []

    if inputs.requires_grad:
        def dense_inputsBackward(grad):
            return grad @ weights.data.T
        depends_on.append(Dependency(weights, dense_inputsBackward))

    if weights.requires_grad:
        def dense_weightsBackward(grad):
            # print(inputs.data.T @ grad)
            return inputs.data.T @ grad
        depends_on.append(Dependency(inputs, dense_weightsBackward))

    if bias.requires_grad:
        def dense_biasBackward(grad):
            return grad.sum(axis=0)
        depends_on.append(Dependency(bias, dense_biasBackward))
    return Tensor(data, requires_grad, depends_on).no_leaf()


# TODO: Dropout 没有考虑对于多维的处理
"""
思路： 先将其reshape成(-1, shape[-1]) 再进行mask最后还原回去
"""


def dropout_1d(inputs: Tensor, keep_prob: float) -> Tensor:
    raise NotImplementedError("BUG")
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


@njit(fastmath=True, parallel=True, cache=True)
def __batch_stride(result, batch, view_shape, strides):
    for i in prange(len(batch)):
        result[i] = np.lib.stride_tricks.as_strided(
            batch[i],
            shape=view_shape,
            strides=strides)


def conv2d(inputs: Tensor, weights: Parameter,
           bias: Parameter = None, stride=1, padding=(0, 0)) -> Tensor:
    """
    inputs -> (Batch Size, Channel, Rows, Cols)
    weights -> (out_channel, in_channel, Rows, Cols)
    bias -> (out_channel, out_x, out_y)
    公式计算卷积后输出的形状
    height_out = (height_in - height_w + 2 * padding) // stride + 1
    width_out = (width_in - width_w + 2 * padding) // stride + 1
    out -> (Batch_size, out_channel, height_out, width_out)
    """
    if weights.shape[1] != inputs.shape[1]:
        raise ValueError(
            f"Weights的in_channel和inputs的不匹配, inputs -> {inputs.shape}, weights -> {weights.shape}")

    # Format Argv
    out_channel = weights.shape[0]
    _inputs = ensure_tensor(inputs)
    weights = ensure_tensor(weights)
    origin_shape = inputs.shape
    if not isinstance(stride, tuple):
        stride = (stride,)*2
    if not isinstance(padding, tuple):
        padding = (padding,)*2


    # Padding
    #####################################################################
    pad_row = (padding[0],)*2  # 上下pad
    pad_col = (padding[1],)*2  # 左右pad
    _inputs.data = np.pad(_inputs.data, [(0, 0), (0, 0), pad_row, pad_col])
    #####################################################################

    if bias is not None:
        if bias.shape[0] == weights.shape[0] and bias.ndim == 1:
            bias = ensure_tensor(bias)
        else:
            raise ValueError(
                f"Bias的形状和输出无法匹配, Expect Shape({weights.shape[0]}) 或\
                    Shape{weights.shape[0], 1, 1}. But receive shape({bias.shape})")

    out_x = (origin_shape[2] - weights.shape[2] + 2 * padding[0]) + 1
    out_y = (origin_shape[3] - weights.shape[3] + 2 * padding[1]) + 1

    requires_grad = inputs.requires_grad or weights.requires_grad or bias.requires_grad

    view_shape = (1, out_x, out_y,
                  weights.shape[1], weights.shape[2], weights.shape[3])
    _strides = _inputs[0].data.strides*2
    batch_result = []

    strided_batch = np.array([np.lib.stride_tricks.as_strided(
        x, view_shape, _strides) for x in _inputs.data])[:, 0, 0:out_x:stride[0], 0:out_y:stride[1]]

    # Numba
    ############################################################################
    # strided_batch = np.empty((origin_shape[0],1,*view_shape[1:]))            #
    # batch_length = origin_shape[0]                                           #
    # __batch_stride(strided_batch, inputs.data, view_shape,_strides)          #
    # strided_batch = strided_batch[:,0,0:out_x:stride[0],0:out_y:stride[1]]   #
    ############################################################################

    # TODO 将 C代码改成Batch Conv2d 看能否提升效率
    for arr in strided_batch:
        batch_result.append(
            c_func.sample_conv2d(arr, weights.data.reshape(
                1, *weights.shape))
        )
    data = np.array(batch_result) 
    # 还没有加上bias
    depends_on: List[Dependency] = []

    # inputs:
    if inputs.requires_grad:
        def conv2d_inputsBackward(grad: np.ndarray) -> np.ndarray:
            inputs_grad = np.ones_like(inputs)
            return grad * inputs_grad
        depends_on.append(Dependency(inputs, conv2d_inputsBackward))

    # weights:
    if weights.requires_grad:
        def conv2d_weightsBackward(grad: np.ndarray) -> np.ndarray:
            weights_grad = np.tile(
                strided_batch.sum(axis=0).sum(axis=0).sum(axis=0),
                (out_channel, 1, 1, 1)
            )
            grad = grad.mean(axis=0).mean(axis=1).mean(axis=1).reshape(out_channel,1,1,1)
            return grad * weights_grad
        depends_on.append(Dependency(weights, conv2d_weightsBackward))

    _tensor = Tensor(data, requires_grad, depends_on).no_leaf()

    # bias:
    if bias is not None:
        return _tensor + bias.reshape(1,*bias.shape,1,1)
    else:
        return _tensor
