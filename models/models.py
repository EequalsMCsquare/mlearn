import numpy as np
from ..layers import Module
from ..optimizers import Optimizer
from ..Loss import *
from typing import List, Union, Callable
from ..utils import DataLoader
import sys

criterionable = Union[Callable, Loss]


class Seq(Module):
    def __init__(self, layers: List = []):
        self.layers = layers
        self.optimizer = None
        self.criterion = None
        self.callbacks = []
        self.summary = "\n"
        layer_iter = iter(self.layers)
        self.summary.join(next(layer_iter))

    def add(self, layer: Module):
        self.layers.append(layer)

    def compile(self, optimizer: Optimizer, criterion: Loss):
        self.optimizer = optimizer
        self.criterion = criterion

    def __repr__(self):
        return self.summary

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
                print(f"\r{epoch}/{EPOCHS} Batch %-4d/{dataset.batch_size}  [{bar}] -> Loss %.5f"%(i,loss.data), end="")
                sys.stdout.flush()
                self.optimizer.step()
            print()

