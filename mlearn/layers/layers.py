from ..autograd.tensor import Tensor
from ..autograd.parameter import Parameter
from .module import Module
from ..functional import layers as F
from ..functional import activation as A


class Dense(Module):
    def __init__(self, input_shape: int, output_shape: int) -> Tensor:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.w = Parameter(input_shape, output_shape)
        self.b = Parameter(output_shape)

    def forward(self, inputs):
        assert (len(inputs.shape) == 2), "必须传入(batch_size,features)"
        return F.dense(inputs, self.w, self.b)

    def __repr__(self):
        return f"全连接层: (Batch Size,{self.input_shape}) -> (Batch Size,{self.output_shape})"

    def __call__(self, inputs):
        return self.forward(inputs)


class Conv2d(Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: [int, tuple],
                 stride:int=1, padding:int=0, padding_mode:str='zeros'):
        self.in_channel = in_channel
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 2:
                raise ValueError("卷积核必须是二维的")
            else:
                self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        self.w = Parameter(self.out_channel, self.in_channel,*self.kernel_size)
        self.b = Parameter(self.out_channel)

    def forward(self, inputs):
        # ( Batch_size, in_channel, x, y )
        if inputs.ndim != 4:
            raise ValueError(f"输入数据形状异常, expect (batch_size, {self.in_channel}, x, y), but receive  {inputs.shape} ")
        if inputs.shape[1] != self.in_channel:
            raise ValueError(f"输入数据的Channel有误, expect {self.in_channel}, but receive {inputs.shape[1]}")
        return F.conv_2d(inputs, self.w, self.b)

    def __repr__(self):
        return f"二维卷积"

    def __call__(self, inputs):
        self.forward(inputs)


class Dropout:
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def forward(self, inputs):
        return F.dropout_1d(inputs, self.keep_prob)

    def __repr__(self):
        return f"Dropout层, 保留特征比例: {self.keep_prob}"

    def __call__(self, inputs):
        return self.forward(inputs)


class Dropout2d:
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def forward(self, inputs):
        return F.dropout_2d(inputs, self.keep_prob)

    def __repr__(self):
        return f"2D Dropout层, 保留特征比例: {self.keep_prob}"

    def __call__(self, inputs):
        return self.forward(inputs)


class ReLU:
    def __init__(self):
        pass

    def __call__(self, inputs):
        return A.relu(inputs)


class Flatten():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inputs: Tensor) -> Tensor:
        return F.flatten(inputs)

    def __repr__(self):
        return "降维打击！！\n"


class Tanh:
    def __init__(self):
        pass

    def __repr__(self):
        return "Tanh层\n"

    def __call__(self, inputs):
        return A.tanh(inputs)
