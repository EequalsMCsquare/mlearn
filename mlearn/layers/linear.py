from ..autograd.parameter import Parameter, param_init
from ..functional import layers as F
from .module import Module
from ..autograd.tensor import Tensor


class Dense(Module):
    def __init__(self, input_shape: int, output_shape: int) -> Tensor:
        super(Dense, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = Parameter(input_shape, output_shape) * \
            param_init.xavier_init(self.input_shape, self.output_shape)
        self.bias = Parameter(output_shape, bias=True)

    def forward(self, inputs):
        assert (len(inputs.shape) == 2), "必须传入(batch_size,features)"
        return F.dense(inputs, self.weights, self.bias)

    def __repr__(self):
        return f"全连接层 -> ( Batch Size, {self.input_shape}) => (Batch Size, {self.output_shape} )"

    def __call__(self, inputs):
        return self.forward(inputs)
