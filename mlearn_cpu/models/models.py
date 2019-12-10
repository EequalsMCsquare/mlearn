import numpy as np
from ..layers import Module
from ..optimizers import Optimizer
from ..Loss import *
from typing import List, Union, Callable,Iterator
from ..utils import DataLoader
from ..autograd.parameter import Parameter
import sys

criterionable = Union[Callable, Loss]



class Model:
    def __init__(self, layers: List = [Module]):
        super().__init__()
        raise NotImplementedError("正确率不对")
        self.layers: List[Module] = layers
        self.trainFeature_shape = None
        self.optimizer = None
        self.criterion = None

    def parameters(self) -> Iterator[Parameter]:
        for layer in self.layers:
            if isinstance(layer, Module):
                yield from layer.parameters()

    def compile(self, optimizer: Optimizer, criterion: Loss):
        self.optimizer = optimizer
        self.criterion = criterion

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer,Module):
                layer.zero_grad()



class Seq(Model):
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
        super().__init__(layers)
        self.callbacks = []


    def add(self, layer: Module):
        self.layers.append(layer)

    def __repr__(self):
        summary = ""
        for layer in self.layers:
            summary += layer.__repr__()
        return summary

    def fit(self, dataset: DataLoader, EPOCHS: int, record_rate=5) -> List:
        assert self.optimizer is not None, "先compile(优化器,误差器),再训练网络"
        assert self.criterion is not None, "先compile(优化器,误差器),再训练网络"

        history = []
        bar = " "*20
        self.trainFeature_shape = dataset.feature_shape

        for epoch in range(EPOCHS):
            running_loss = 0
            for i, batch in enumerate(dataset):
                features, labels = batch
                self.zero_grad()
                out = features
                for layer in self.layers:
                    out = layer(out)
                loss = self.criterion(out, labels)
                loss.backward()
                print(f"\r{epoch+1}/{EPOCHS} Batch %-4d/{dataset.total_batch}  [{bar}] -> Loss %.5f"%(i,loss.data/record_rate), end="")
                sys.stdout.flush()
                self.optimizer.step()
                running_loss += loss.data
            history.append((running_loss/dataset.total_batch).tolist())
                              
            print()
        print("Trainning Complete!")
        return history


    def predict(self, dataset:DataLoader) -> List:
        if self.trainFeature_shape != dataset.feature_shape:
            raise ValueError(f"测试集和训练集的形状不匹配,Expect {self.trainFeature_shape}, But Receive {dataset.feature_shape}")

        correct = 0
        total = 0
        for batch in dataset:
            features,labels = batch
        out = features
        for layer in self.layers:
            out = layer(out)
        predict = []
        for x in out.data:
            predict.append(np.argmax(x))
        for b in predict == labels.data:
            if b:
                correct += 1
            total += dataset.batch_size
        print(correct)
        print(total)
        print("Accuray: %.5f"%(correct / total))