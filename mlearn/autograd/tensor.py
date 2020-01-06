from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

# No scientific notation
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=80)

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, int, np.ndarray]


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
        self.__data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad: Optional['Tensor'] = None
        self.ndim = self.data.ndim

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self.__data

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return (self.__data.dtype)

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        # if isinstance(new_data,Tensor):
        #     new_data = new_data.data
        self.__data = new_data
        self.grad = None

    def __setGrad(self, grad: np.ndarray) -> None:
        self.grad = grad

    # Transpose
    def transpose(self) -> 'Tensor':
        _data = self.data.T
        requires_grad = self.requires_grad
        if requires_grad:
            _grad = self.grad.T
            depends_on = self.depends_on
        else:
            _grad = None
            depends_on = []
        _tensor = Tensor(_data, requires_grad, depends_on)
        _tensor.__setGrad(_grad)
        return _tensor

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
        data = self.data.astype(np.float32)
        requires_grad = self.requires_grad
        depends_on = self.depends_on
        return Tensor(data, requires_grad, depends_on)

    def double(self) -> 'Tensor':
        self.data.astype(np.float64)
        return self

    def long(self) -> 'Tensor':
        data = self.data.astype(np.long)
        requires_grad = self.requires_grad
        depends_on = self.depends_on
        return Tensor(data, requires_grad, depends_on)

    def reshape(self, *shape) -> 'Tensor':
        data = self.data.reshape(*shape)
        requires_grad = self.requires_grad
        if requires_grad:
            def reshapeBackward(grad: np.ndarray) -> np.ndarray:
                return grad.reshape(*self.grad.shape)
            depends_on = [Dependency(self, reshapeBackward)]
        else:
            depends_on = []

        return Tensor(data,
                      requires_grad,
                      depends_on)

    def flatten(self) -> 'Tensor':
        return Tensor(self.data.flatten(),
                      self.requires_grad,
                      self.depends_on)

    def __repr__(self) -> str:
        if self.requires_grad:
            if len(self.depends_on) != 0:
                return f"Tensor({np.array2string(self.data,prefix='Tensor(',separator=', ', sign='-',floatmode='maxprec_equal',precision=4)}, grad_fn=<{self.depends_on[0].grad_fn.__name__}>)"
            else:
                return f"Tensor({np.array2string(self.data,prefix='Tensor(',separator=', ', sign='-',floatmode='maxprec_equal',precision=4)}, requires_grad={self.requires_grad})"

        else:
            return f"Tensor({np.array2string(self.data,prefix='Tensor(', separator=', ', sign='-',floatmode='maxprec_equal',precision=4)})"

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

    def __rtruediv__(self, var) -> 'Tensor':
        return _div(ensure_tensor(var), self)

    def __pow__(self, var) -> 'Tensor':
        if not isinstance(var, (int, float)):
            raise TypeError("次方只能是整型或浮点型")
        return _pow(self, var)

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
        def sumBackward(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor,
            so each input element contribute that much.
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, sumBackward)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def zeros(*shape, requires_grad: bool = False) -> 'Tensor':
    data = np.zeros(shape)
    return Tensor(data, requires_grad)


def zeros_like(tensor, requires_grad: bool = False) -> 'Tensor':
    data = np.zeros(tensor.shape)
    return Tensor(data, requires_grad)


def ones(*shape, requires_grad: bool = False) -> 'Tensor':
    return Tensor(np.ones(shape), requires_grad)


def randn(*shape, requires_grad: bool = False) -> 'Tensor':
    return Tensor(np.random.randn(*shape), requires_grad)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []
    if t1.requires_grad:
        def addBackward1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, addBackward1))

    if t2.requires_grad:
        def addBackward2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, addBackward2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def mulBackward1(grad: np.ndarray) -> np.ndarray:
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

        depends_on.append(Dependency(t1, mulBackward1))
    if t2.requires_grad:
        def mulBackward2(grad: np.ndarray) -> np.ndarray:
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

        depends_on.append(Dependency(t2, mulBackward2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _pow(t: Tensor, pow: float) -> Tensor:
    data = t.data**pow
    requires_grad = t.requires_grad
    if requires_grad:
        def pow_backward(grad: np.ndarray) -> np.ndarray:
            return grad * (t.data**(pow-1))*pow
        depends_on = [Dependency(t, pow_backward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def matmulBackward1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T
        depends_on.append(Dependency(t1, matmulBackward1))

    if t2.requires_grad:
        def matmulBackward2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2, matmulBackward2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        def negBackward(grad: np.ndarray) -> np.ndarray:
            return -grad
        depends_on = [Dependency(t, negBackward)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def _slice(t: Tensor, idx: slice) -> Tensor:
    data = t.data[idx]
    requires_grad = t.requires_grad
    if requires_grad:
        # BUG
        def sliceBackward(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idx] = grad
            return bigger_grad

        depends_on = Dependency(t, sliceBackward)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _div(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def divBackward1(grad: np.ndarray) -> np.ndarray:
            grad = grad / t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, divBackward1))
    if t2.requires_grad:
        def divBackward2(grad: np.ndarray) -> np.ndarray:
            grad = grad / t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, divBackward2))

    return Tensor(data,
                  requires_grad,
                  depends_on)
