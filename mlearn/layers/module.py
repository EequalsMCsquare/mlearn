import inspect
from typing import Iterator
from collections import OrderedDict

from ..autograd.tensor import Tensor
from ..autograd.parameter import Parameter


class Module:
    def __init__(self,):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

    def forward(self, inputs):
        raise NotImplementedError

    def __track_parameters(self, name, value):
        if value is None:
            self._parameters[name] = None
        elif not isinstance(value, Parameter):
            raise TypeError(f"不能將一個{type(value)}指定爲Parameter\
            (必須是mlearn.autograd.Parameter 或 None)")
        else:
            self._parameters[name] = value

    def __add_modules(self, name, value):
        if value is None:
            self._modules[name] = None
        elif not isinstance(value, Module):
            raise TypeError(f"不能將一個{type(value)}指定爲Module\
            (必須是mlearn.layers.Module 或 None)")
        else:
            self._modules[name] = value

    def parameters(self) -> Iterator[Parameter]:
        if len(self._modules) == 0:
            for k, v in self._parameters.items():
                if isinstance(v, Parameter):
                    yield v
        elif len(self._parameters) == 0:
            for k, v in self._modules.items():
                if isinstance(v, Module):
                    yield from v._parameters.values()

    # def parameters(self) -> Iterator[Parameter]:
    #     for name, value in inspect.getmembers(self):
    #         if isinstance(value, Parameter):
    #             yield value
    #         elif isinstance(value, Module):
    #             yield from value.parameters()

    def __setattr__(self, name, value):
        if not isinstance(name, str):
            raise TypeError(f"Attribute的名字必須是 <str>， 但獲得 <{type(name)}>")
        elif '.' in name:
            raise KeyError("Attribute的名字不能包含 \".\"")
        elif name == ' ':
            raise KeyError("Attribute的名字不能空")
        else:
            if isinstance(value, Parameter):
                if self._parameters is None:
                    raise AttributeError("先調用基類的構造器 -> super().__init__()")
                self.__track_parameters(name, value)
            elif isinstance(value, Module):
                if self._modules is None:
                    raise AttributeError("先調用基類的構造器 -> super().__init__()")
                self.__add_modules(name, value)
            else:
                object.__setattr__(self, name, value)

    def __getattr__(self, name:str):
        if len(self._parameters) == 0:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        if len(self._modules) == 0:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, inputs):
        return self.forward(inputs)

    def load_wb(self, PATH: str) -> None:
        """
        读取训练好的模型的Weights n' Bias
        """
        raise NotImplementedError("读取训练好的模型的Weights n' Bias")
