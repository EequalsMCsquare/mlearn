from autograd.tensor import Tensor
from autograd.parameter import Parameter
from autograd.module import Module


class Dense(Module):
    def __init__(self, input_shape: int, output_shape: int) -> Tensor:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.w = Parameter(input_shape, output_shape)
        self.b = Parameter(output_shape)

    def forward(self, inputs):
        assert len(inputs.shape) == 2, "必须传入(batch_size,features)"
        return inputs @ self.w + self.b

    def __repr__(self):
        return f"全连接层\n输入形状->(Batch Size,{self.input_shape})\
            输出形状->(Batch Size,{self.output_shape})"

class Dropout(Module):
    pass
