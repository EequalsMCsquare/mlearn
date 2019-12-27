import mlearn as mlearn
from mlearn import functional as F
from mlearn import layers
from mlearn.optimizers import SGD, RMSProp, Momentum
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from mlearn.utils import DataLoader
from mlearn.utils import pre_F as P
import sys

train = datasets.MNIST("datasets", train=True, download=True)

test = datasets.MNIST("datasets", train=False, download=True)

pre = [P.normalize_MinMax]
trainset = DataLoader((train.data,train.targets),batch_size=32,shuffle=True,
                      preprocessing=pre)
testset = DataLoader((test.data, test.targets), batch_size=32, shuffle=True,
                    preprocessing=pre)


class Net(mlearn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.dense1 = layers.Dense(784,300)
        self.dense2 = layers.Dense(300,150)
        self.dense3 = layers.Dense(150,64)
        self.dense4 = layers.Dense(64,10)
        
        
    def forward(self, inputs):
        o = self.dense1(inputs)
        o = F.relu(o)
        o = self.dense2(o)
        o = F.relu(o)
        o = self.dense3(o)
        o = F.relu(o)
        o = self.dense4(o)
        o = F.relu(o)
        return o
net = Net()


def fit():
    hist = []
    optimizer = RMSProp(net,0.001)
    EPOCHS = 10
    net.zero_grad()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        bar = " "*20
        for i, batch in enumerate(trainset, 0):
            
            features, labels = batch
            net.zero_grad()
            predict = net(features.reshape(-1,784))
            loss = F.cross_entropy(predict, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            print(f"\r{epoch+1}/{EPOCHS} Batch %-4d/1874  [{bar}] -> Loss %.5f"%(i,loss.data), end="")
            sys.stdout.flush()
        print()
        hist.append(loss.data)
    return np.array(hist)
    
print('trainning completed!')
hist = fit()