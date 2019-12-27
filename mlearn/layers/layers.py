from ..autograd.tensor import Tensor
from ..autograd.parameter import Parameter
from .module import Module
from ..functional import layers as F
from ..functional import activation as A


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
