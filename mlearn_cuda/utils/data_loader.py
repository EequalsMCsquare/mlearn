import numpy as cp
from ..autograd import tensor
from .pre_F import *
from typing import Tuple, Callable, List
from ..autograd.tensor import ensure_array


class DataLoader:
    def __init__(self, dataset: Tuple, batch_size: int = 32, shuffle: bool = False, preprocessing: List = [], ToTensor: bool = True):
        assert isinstance(dataset, Tuple), "数据集必须是一个Tuple->(训练集,测试集)"
        if ensure_array(dataset[0]).shape[0] != ensure_array(dataset[1]).shape[0]:
            raise ValueError("数据集的特征的样本数量和标签的样本数量不匹配")
        self.preprocessing = preprocessing
        self.features, self.labels = \
            data_shuffle(dataset[0], dataset[1]) if shuffle else dataset
        self.batch_size = batch_size
        self.ToTensor = ToTensor
        self.feature_shape = self.features.shape[1:]
        self.transform()

    @property
    def length(self):
        return round(self.features.shape[0]/self.batch_size)

    def transform(self):
        for pre_fn in self.preprocessing:
            assert isinstance(pre_fn, Callable), "必须传入函数指针"
            self.features, self.labels = pre_fn(self.features, self.labels)

    def batch_generator(self) -> cp.ndarray:
        _dataset = []
        _len = len(self.labels)
        for i in range(0, _len, self.batch_size):
            _dataset.append(
                toTensor(self.features[i:i+self.batch_size],self.labels[i:i+self.batch_size])  
            ) if toTensor else (
                    (self.features[i:i+self.batch_size],self.labels[i:i+self.batch_size])
            )
        self.total_batch = len(_dataset)
        return _dataset

    def __iter__(self):
        return iter(self.batch_generator())

    def __repr__(self):
        return f"总共有 {self.length} Batch\n特征集形状->{self.features.shape}\n标签集形状->{self.labels.shape}"