import cupy as cp
from typing import NamedTuple, Callable, Union, Optional, List


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable = [[cp.ndarray], cp.ndarray]


Arrayable = Union[float, list, cp.ndarray]
Tensorable = Union[list, 'Tensor', cp.ndarray]


def ensure_array(arrayable: Arrayable) -> cp.ndarray:
    if isinstance(arrayable, cp.ndarray):
        return arrayable
    else:
        return cp.asarray(arrayable)


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor(object):
    def __init__(self, data: cp.ndarray,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        super().__init__()
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []


        self.shape = self.data.shape
        self.device = self._data.device
        self.dtype = self._data.dtype

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> cp.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: cp.ndarray) -> None:
        if isinstance(new_data, Tensor):
            new_data = new_data.data
        self._data = new_data
        self.grad = None

    def reshape(self, *shape) -> 'Tensor':
        data = self.data.reshape(*shape)
        if self.requires_grad and self.grad is not None:
            self.grad.reshape(*shape)
        else:
            grad = None
        return self
    def reshape(self, *shape) -> 'Tensor':
        return Tensor(self.data.reshape(shape),
                      self.requires_grad,
                      self.depends_on)

    def long(self) -> 'Tensor':
        return Tensor(self.data.astype(cp.int16),self.requires_grad,self.depends_on,self.grad)

    def __repr__(self):
        return f"{self.data} requires_grad={self.requires_grad}" if self.requires_grad else\
            f"{self.data}"

    def zero_grad(self):
        self.grad = Tensor(cp.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "在无梯度追踪要求的Tensor上调用backward()"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.)
            else:
                raise ValueError("只能够对Scaler进行求导")
        self.grad.data = self.grad.data + grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

    def __add__(self, var) -> 'Tensor':
        return _add(self, ensure_tensor(var))

    def __radd__(self, var) -> 'Tensor':
        return _add(ensure_tensor(var), self)

    def __iadd__(self, var) -> 'Tensor':
        if self.requires_grad:
            raise AttributeError("不能对追踪梯度的Tensor进行自增")
        self.data = self.data + ensure_tensor(var).data
        return self

    def __neg__(self):
        return _neg(self)

    def __sub__(self, var) -> 'Tensor':
        return _sub(self, ensure_tensor(var))

    def __rsub__(self, var) -> 'Tensor':
        return _sub(ensure_tensor(var), self)

    def __isub__(self, var) -> 'Tensor':
        # if self.requires_grad:
        #     raise AttributeError("不能对追踪梯度的Tensor进行自减")
        self.data = self.data - ensure_tensor(var).data
        return self

    def __mul__(self, var) -> 'Tensor':
        return _mul(self, ensure_tensor(var))

    def __rmul__(self, var) -> 'Tensor':
        return _mul(ensure_tensor(var), self)

    def __imul__(self, var) -> 'Tensor':
        if self.requires_grad:
            raise AttributeError("不能对追踪梯度的Tensor进行自乘")
        self.data = self.data * ensure_tensor(var).data
        return self

    def __truediv__(self, var) -> 'Tensor':
        return _div(self, ensure_tensor(var))

    def __rtruediv__(self, var) -> 'Tensor':
        return _div(ensure_tensor(var), self)

    def __itruediv__(self, var) -> 'Tensor':
        if self.requires_grad:
            raise AttributeError("不能对追踪梯度的Tensor进行自除")
        self.data = self.data / ensure_tensor(var).data
        return self

    def __matmul__(self, var) -> 'Tensor':
        assert(isinstance(var, Tensor)), "只能Tensor 和 Tensor进行矩阵乘法"
        return _matmul(self, var)

    def __pow__(self, var) -> 'Tensor':
        if not isinstance(var, (int, float)):
            raise TypeError("次数只能是整型或浮点")
        return _pow(self, var)

    def __getitem__(self, index) -> 'Tensor':
        return _slice(self, index)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(
        cp.random.randn(*shape),
        requires_grad
    )


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(
        cp.zeros(shape),
        requires_grad
    )


def ones(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(
        cp.zeros(shape),
        requires_grad
    )


def zeros_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:

    return Tensor(
        cp.zeros(tensor.shape),
        requires_grad
    )


def ones_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("只能接受Tensor类型")
    return Tensor(
        cp.ones_like(tensor.data),
        requires_grad
    )


def _tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def sum_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * cp.ones_like(t.data)
        depends_on = [Dependency(t, sum_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def add_backward(grad: cp.ndarray) -> cp.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, add_backward))

    if t2.requires_grad:
        def add_backward(grad: cp.ndarray) -> cp.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdim=True)
            return grad
        depends_on.append(Dependency(t2, add_backward))
    return Tensor(data, requires_grad, depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad

    if requires_grad:
        def neg_backward(grad: cp.ndarray) -> cp.ndarray:
            return -grad
        depends_on = [Dependency(t, neg_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def mul_backward(grad: cp.ndarray) -> cp.ndarray:
            grad = grad * t2.data
            ndims_add = grad.ndim - t1.data.ndim
            for _ in range(ndims_add):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=1, keepdim=True)
            return grad
        depends_on.append(Dependency(t1, mul_backward))

    if t2.requires_grad:
        def mul_backward(grad: cp.ndarray) -> cp.ndarray:
            grad = grad * t1.data
            ndims_add = grad.ndim - t2.data.ndim
            for _ in range(ndims_add):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=1, keepdim=True)
            return grad
        depends_on.append(Dependency(t2, mul_backward))

    return Tensor(data, requires_grad, depends_on)


def _div(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def div_backward(grad: cp.ndarray) -> cp.ndarray:
            grad = grad / t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, div_backward))
    if t2.requires_grad:
        def div_backward(grad: cp.ndarray) -> cp.ndarray:
            grad = grad / t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, div_backward))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def matmul_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad @ t2.data.T
        depends_on.append(Dependency(t1, matmul_backward))

    if t2.requires_grad:
        def matmul_backward(grad: cp.ndarray) -> cp.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2, matmul_backward))

    return Tensor(data, requires_grad, depends_on)


def _pow(tensor: Tensor, pow: Union[int, float]) -> Tensor:
    data = tensor.data ** pow
    requires_grad = tensor.requires_grad

    if requires_grad:
        def pow_backward(grad: cp.ndarray) -> cp.ndarray:
            return grad * ((tensor.data**(pow-1))*pow)
        depends_on = [Dependency(tensor, pow_backward)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _slice(t: Tensor, idx: slice) -> Tensor:
    data = t.data[idx]
    requires_grad = t.requires_grad

    if requires_grad:
        def slice_backward(grad: cp.ndarray) -> cp.ndarray:
            bigger_grad = cp.zeros_like(data)
            bigger_grad[idx] = grad
            return bigger_grad

        depends_on = Dependency(t, slice_backward)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)