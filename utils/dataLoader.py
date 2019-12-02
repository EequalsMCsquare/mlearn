import numpy as np
from ..autograd import tensor
from .pre import *
from typing import Tuple
from ..autograd.tensor import ensure_array


class DataLoader:
    def __init__(self, dataset: Tuple, batch_size=32, shuffle=False, preprocessing=[]):
        assert isinstance(dataset, Tuple), "数据集必须是一个Tuple->(训练集,测试集)"
        if ensure_array(dataset[0]).shape[0] != ensure_array(dataset[1]).shape[0]:
            raise ValueError("数据集的特征的样本数量和标签的样本数量不匹配")
        self.features, self.labels = \
            data_shuffle(dataset[0], dataset[1]) if shuffle else dataset
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.total_batch = None

    def batch_generator(self) -> np.ndarray:
        _dataset = []
        _len = len(self.labels)
        for i in range(0, _len, self.batch_size):
            _dataset.append(
                (self.features[i:i+self.batch_size],
                    self.labels[i:i+self.batch_size])
            )
        self.total_batch = len(_dataset)
        return _dataset


    def __iter__(self):
        return iter(self.batch_generator())

    def __repr__(self):
        return f"总共有 {self.total_batch} Batch\n特征集形状->{self.features.shape}\n标签集形状->{self.labels.shape}"