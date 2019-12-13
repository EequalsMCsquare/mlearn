# MLearn
> A Based on numpy (CPU) & cupy(CUDA) simple machine learning pakage

## Requirements
> Numpy  
> Cupy (only if train on CUDA) 可能会在过后的版本取消。要是有时间的话就用numpy + C + CUDA代替
----
### Features
1. 基于numpy的矩阵运算
2. 自动求道机制
3. 优化器
    - SGD
    - Momentum
    - RMSProp
4. 激活函数
  - ReLU
  - Leaky ReLU
  - Tanh
  - simoid
  - 无梯度计算的softmax `交叉熵误差函数一已经内置了softmax`
5. 误差函数
  - MSE
  - Cross Entropy
6. 数据预处理
  - 标签One-hot
  - 归一化
  - Min Max归一化
  - 0 Mean 归一化
  - 打乱数据集
  - 拆分数据集
  - DataLoader: 数据集分批 -> 迭代器
  
### @ZHONG Xiao
