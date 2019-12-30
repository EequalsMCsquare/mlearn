import inspect
from typing import Iterator
from collections import OrderedDict
import pickle

from ..autograd.tensor import Tensor
from ..autograd.parameter import Parameter
from ..exception import ShapeError, LengthError


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
        if len(self._parameters) == 0 and len(self._modules) == 0:
            raise RuntimeError("既没有_modules 也没有 _parameters你load到哪？哈批.")
        with open(PATH, 'rb') as FILE:
            param = pickle.load(FILE)

        if len(self._modules) == 0:
            """
            1. 如果 self._modules为空的话就说明这是一个曾
            2. 迭代判定是否参数形状匹配
            3. 如果参数形状匹配就直接赋值给self._parameters
            """
            for k, v in self._parameters.items():
                if v.shape != param[k].shape:
                    raise ValueError(f"参数 <{k}> 形状不匹配,保存的模型形<{k}>状是 [{param[k].shape}], 但是当前模型的<{k}>是{v.shape}")
            self._parameters[k] = param[k]
            print("层参数加载完毕")


        elif len(self._parameters) == 0:
            """
                1. 如果 self._parameters 为空的话就说明这是一个神经网络， 里面储存了layers
                2. 判定层数是否相同
                3. 如果层数相同,就开始判定是否参数形状相同
                4. 如果这一层参数相同就直接对层._parameters进行赋值
            """
            if len(self._modules) != len(param):
                raise LengthError("层数无法匹配", len(self._modules), len(param))
            for k, v in self._modules.items():
                for layer_k, layer_v in v._parameters.items():
                    if param[k][layer_k].shape != layer_v.shape:
                        raise ShapeError(f"{k}层的{layer_k}形状不匹配", layer_v.shape, param[k][layer_k].shape)
                self._modules[k]._parameters = param[k]
            print("网络参数加载完毕")


    def save_wb(self, PATH:str, desc:str) -> None:
        if len(self._parameters) == 0 and len(self._modules) == 0:
            raise RuntimeError("既没有_modules 也没有 _parameters你save个啥？哈批.")
        if len(self._parameters) == 0:
            module_state_dict = {}
            for k, v in self._modules.items():
                module_state_dict[k] = v._parameters
            with open(PATH,'wb') as FILE:
                pickle.dump(module_state_dict,FILE,protocol=pickle.HIGHEST_PROTOCOL)
            print("网络参数保存成功")
        elif len(self._modules) == 0:
            with open(PATH, 'wb') as FILE:
                pickle.dump(self._parameters,FILE,protocal=pickle.HIGHEST_PROTOCOL)
            print("层参数保存成功")
