import unittest

from autograd.tensor import Tensor, add

class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = Tensor([4,5,6], requires_grad=True)

        t3 = add(t1,t2)
        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-1,-2,-3]
        assert t2.grad.data.tolist() == [-1,-2,-3]
