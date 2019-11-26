from autograd.tensor import Tensor
import numpy as np

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, 3, -2]))
y_data = x_data @ coef + 5

w = Tensor(np.zeros(3), requires_grad=True)
b = Tensor(np.zeros(()), requires_grad=True)

batch_size = 32

learning_rate = 0.001
for epoch in range(100):
    for start in range(0, 100, batch_size):
        end = start + batch_size
        w.zero_grad()
        b.zero_grad()
        inputs = x_data[start:end]

        predicted = inputs @ w + b
        target = y_data[start:end]
        errors = predicted - target
        loss = (errors * errors).sum()
        loss.backward()
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
    
    if epoch % 20 == 19:
        print(epoch, loss)
