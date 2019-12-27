from ..autograd.tensor import Tensor
from ..autograd.parameter import Parameter
from .module import Module
from ..functional import layers as F
from typing import Tuple, Union


iskernelsize = Union[int, Tuple]

def ensure_tuple(inputs:iskernelsize):
    if isinstance(inputs, tuple):
        return inputs
    else:
        return (inputs,)

class _ConvNd(Module):
    def __init__(self, in_channel:int, out_channel:int, kernel_size:iskernelsize,
                stride:int, padding:int, padding_mode:str):
        super(_ConvNd,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.weights = Parameter(self.out_channel, self.in_channel, *self.kernel_size)
        self.bias = Parameter(self.out_channel)

class Conv2d(_ConvNd):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: iskernelsize,
                 stride:int=1, padding:int=0, padding_mode:str='zeros'):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)*2
        super(Conv2d,self).__init__(
        in_channel, out_channel, kernel_size, stride, padding, padding_mode)

    def forward(self, inputs):
        # ( Batch_size, in_channel, x, y )
        if inputs.ndim != 4:
            raise ValueError(f"输入数据形状异常, expect (batch_size, {self.in_channel}, x, y), but receive  {inputs.shape} ")
        if inputs.shape[1] != self.in_channel:
            raise ValueError(f"输入数据的Channel有误, expect {self.in_channel}, but receive {inputs.shape[1]}")
        return F.conv_2d(inputs, self.w, self.b)

    def __repr__(self):
        return f"二维卷积Object -> ( {self.in_channel}, {self.out_channel}, kernel_size={self.kernel_size}, stride={self.stride} ) "

    def __call__(self, inputs):
        self.forward(inputs)
