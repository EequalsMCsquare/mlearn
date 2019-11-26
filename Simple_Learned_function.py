from autograd import Tensor, Parameter, Module
import numpy as np

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, 3, -2]))
y_data = x_data @ coef + 5


class MyModel(Module):
    def __init__(self) -> None:
        self.w = Parameter(3)
        self.b = Parameter()

    def predict(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b


batch_size = 32
learning_rate = 0.001
model = MyModel()

for epoch in range(100):
    for start in range(0, 100, batch_size):
        end = start + batch_size
        model.zero_grad()
        inputs = x_data[start:end]

        predicted = model.predict(inputs)
        target = y_data[start:end]
        errors = predicted - target
        loss = (errors * errors).sum()
        loss.backward()
        model.w -= model.w.grad * learning_rate
        model.b -= model.b.grad * learning_rate

    if epoch % 20 == 19:
        print(epoch, loss)
