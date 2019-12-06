from ..layers import Module
from ..optimizer import Optimizer
from ..loss import *
from typing import List, Union, Callable
from ..utils import DataLoader
import sys

criterionable = Union[Callable, Loss]


class Seq(Module):
    """
    序列式模型
    参数: layers: List

    方法:
    self.add(神经网络层)
    向layers里添加一个层
    ---
    self.compile(优化器，误差函数)
    为神经网络定义优化器 和 误差函数
    ---
    self.fit(数据集(DataLoader类), 训练次数)
    使用数据集来训练神经网络
    """
    def __init__(self, layers: List = []):
        self.layers = layers
        self.optimizer = None
        self.criterion = None
        self.callbacks = []

    def add(self, layer: Module):
        self.layers.append(layer)

    def compile(self, optimizer: Optimizer, criterion: Loss):
        self.optimizer = optimizer
        self.criterion = criterion

    def __repr__(self):
        summary = ""
        for layer in self.layers:
            summary += layer.__repr__()
        return summary
        
    def fit(self, dataset: DataLoader, EPOCHS: int):
        assert self.optimizer is not None, "先compile(优化器,误差器),再训练网络"
        assert self.criterion is not None, "先compile(优化器,误差器),再训练网络"
        bar = " "*20
        for epoch in range(EPOCHS):
            for i, batch in enumerate(dataset):
                features, labels = batch
                out = features
                for layer in self.layers:
                    out = layer(out)
                loss = self.criterion(out,labels)
                loss.backward()
                print(f"\r{epoch+1}/{EPOCHS} Batch %-4d/{dataset.total_batch}  [{bar}] -> Loss %.5f"%(i,loss.data), end="")
                sys.stdout.flush()
                self.optimizer.step()
            print()
        print("Trainning Complete!")
        