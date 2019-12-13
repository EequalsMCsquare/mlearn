from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

np.set_printoptions(
    suppress=True,
    precision=6,
    formatter={'float': '{:0.4f}'.format}
)


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.array]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:

        if isinstance(new_data,Tensor):
            new_data = new_data.data
        self._data = new_data
        self.grad = None

    def numpy(self) -> np.ndarray:
        return self.data

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "在无梯度记录要求的Tensor上调用backward()"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("只能够对Scaler进行求导")

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

    def float(self) -> 'Tensor':
        data = self.data.astype(np.float64)
        requires_grad = self.requires_grad
        depends_on = self.depends_on
        return Tensor(data, requires_grad, depends_on)

    def long(self) -> 'Tensor':
        data = self.data.astype(np.long)
        requires_grad = self.requires_grad
        depends_on = self.depends_on
        return Tensor(data, requires_grad, depends_on)

    def reshape(self, *shape) -> 'Tensor':
        return Tensor(self.data.reshape(shape),
                      self.requires_grad,
                      self.depends_on)

    def flatten(self) -> 'Tensor':
        return Tensor(self.data.flatten(),
                      self.requires_grad,
                      self.depends_on)

    def __repr__(self) -> str:
        if self.requires_grad:
            return f"形状->{self.shape}    梯度 [有]\n{repr(self.data).replace('array','Tnsor')}"
        else:
            return f"{repr(self.data).replace('array','Tnsor')}"

    """
    ####################### Operators ###########################
    """

    def __add__(self, var) -> 'Tensor':
        # self + var
        return _add(self, ensure_tensor(var))

    def __radd__(self, var) -> 'Tensor':
        # var + self
        return _add(ensure_tensor(var), self)

    def __iadd__(self, var) -> 'Tensor':
        self.data = self.data + ensure_tensor(var).data
        # self.grad = None
        return self

    def __mul__(self, var) -> 'Tensor':
        return _mul(self, ensure_tensor(var))

    def __rmul__(self, var) -> 'Tensor':
        return _mul(ensure_tensor(var), self)

    def __imul__(self, var) -> 'Tensor':
        self.data = self.data * ensure_tensor(var).data
        #self.grad = None
        return self

    def __truediv__(self, var) -> 'Tensor':
        return _div(self, ensure_tensor(var))

    def __rtruediv__(self,var) -> 'Tensor':
        return _div(ensure_tensor(var),self)

    def __pow__(self, var) -> 'Tensor':
        tensor = self
        for _ in range(var-1):
            tensor = tensor * self
        return tensor

    def __matmul__(self, var) -> 'Tensor':
        return _matmul(self, var)

    def __sub__(self, var) -> 'Tensor':
        return _sub(self, ensure_tensor(var))

    def __rsub__(self, var) -> 'Tensor':
        return _sub(ensure_tensor(var), self)

    def __isub__(self, var) -> 'Tensor':
        self.data = self.data - ensure_tensor(var).data
        return self

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def __neg__(self) -> 'Tensor':
        return _neg(self)


def _tensor_sum(t: Tensor) -> Tensor:
    """
    Take a tensor and returns the 0-tensor 
    that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor,
            so each input element contribute that much.
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)



def zeros(*shape, requires_grad: bool = False) -> 'Tensor':
    data = np.zeros(shape)
    return Tensor(data, requires_grad)

def zeros_like(tensor,requires_grad:bool = False) -> 'Tensor':
    data = np.zeros(tensor.shape)
    return Tensor(data,requires_grad)


def ones(*shape, requires_grad: bool = False) -> 'Tensor':
    return Tensor(np.ones(shape), requires_grad)


def randn(*shape, requires_grad: bool = False) -> 'Tensor':
    return Tensor(np.random.randn(*shape), requires_grad)







def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def _slice(t: Tensor, idx: slice) -> Tensor:
    data = t.data[idx]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idx] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _div(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad / t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad / t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)