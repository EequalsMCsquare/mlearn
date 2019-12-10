from ..autograd.tensor import Tensor, Dependency, ensure_tensor,zeros
from ..autograd.parameter import Parameter

import numpy as np
np.set_printoptions(
    suppress=True,
    precision=3,
    formatter={'float': '{:0.3f}'.format}
)

"""
################### Layers #######################
"""


def dense(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    #assert (inputs.shape[-1] == weights.shape[0]), "傻逼，[形状不匹配]"
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


def conv_2d(inputs: Tensor, weights: Parameter, bias:Parameter = None, stride: int = 1, padding:int = 0) -> Tensor:
    """
    inputs -> (Batch Size, Channel, Rows, Cols)
    weights -> (out_channel, in_channel, Rows, Cols)
    bias -> (Rows, Cols)
    outputs -> (Batch Size, out_channel,inputs_rows - kernel_rows + stride,inputs_cols - kernel_cols + stride)
    """
    in_channels = inputs.shape[1]
    inputs_rows = inputs.shape[2]
    inputs_cols = inputs.shape[3]

    kernel_out_channel = weights.shape[0]
    kernel_rows = weights.shape[2]
    kernel_cols = weights.shape[3]
    shape_after_conv = (inputs.shape[0],kernel_out_channel,inputs_rows - kernel_rows + stride,inputs_cols - kernel_cols + stride)
    _data = inputs.data
    final_result = np.array([]) # All Batch
    if bias is None:
        bias = zeros(kernel_rows,kernel_cols)
        
    def sample_conv( batch1_inputs:np.ndarray) -> np.ndarray:
        _result = np.array([]) # out_channel
        for channel in range(kernel_out_channel):
            _tmp_all = np.array([]) # 一面
            for row in range(0,inputs_rows - kernel_rows + stride, stride):
                _tmp = np.array([]) # 一排
                for col in range(0, inputs_cols - kernel_cols + stride, stride):
                    _tmp = np.hstack((
                            _tmp, 
                            (batch1_inputs[:,row:row+kernel_rows,col:col+kernel_cols] * weights[channel].data + bias.data).sum()
                        ))
                _tmp_all = np.hstack((
                    _tmp_all,
                    _tmp
                ))
            _result = np.hstack((
                    _result,
                    _tmp_all
                ))


        return _result

    for batch in _data:
        final_result = np.concatenate((
            final_result, sample_conv(batch)
        ))
    final_result = final_result.reshape(shape_after_conv)

    requires_grad = inputs.requires_grad or weights.requires_grad
    
    if requires_grad:
        def conv2d_backward(grad: np.ndarray) -> np.ndarray:
            return grad
        depends_on = [Dependency(inputs,conv2d_backward)]
    else:
        depends_on = []

    return Tensor(final_result,requires_grad,depends_on)
            