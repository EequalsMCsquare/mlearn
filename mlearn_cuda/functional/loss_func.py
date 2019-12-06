from ..autograd.tensor import Tensor, Dependency, ensure_tensor
import cupy as cp


"""
################### Loss Function #########################
"""


def mse(predicts: Tensor, targets: Tensor) -> Tensor:
    assert (predicts.shape ==
            targets.shape), f"[形状不匹配], 预测->{predicts.shape}但是目标却是{targets.shape}"
    tmp = 1
    for x in predicts.shape:
        tmp *= x
    # result = [data**2 for data in (predicts - targets)]#.sum()/tmp
    result = ((predicts - targets)**2) .sum()/tmp
    # return result

    data = result.data
    requires_grad = result.requires_grad
    if requires_grad:
        depends_on = result.depends_on
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)



def cross_entropy(predicts: Tensor, targets: Tensor) -> Tensor:

    # Stable softmax without grad compute
    def softmax(tensor: Tensor):
        def _stable_softmax(x: cp.ndarray) -> cp.ndarray:
            x = x - cp.max(x)
            _sum = cp.sum(cp.exp(x))
            return cp.exp(x) / _sum
        data = []
        for x in predicts:
            data.append(_stable_softmax(x.data).tolist())
        
        data = cp.asarray(data)
        return Tensor(data, tensor.requires_grad,
                      [Dependency(tensor, lambda x: x)])

    m = targets.shape[0]
    p = softmax(predicts).data
    p[p == 0] = 1e-8 * cp.random.randint(1, 10)
    log_likelihood = -cp.log(p[cp.arange(m), targets.data])
    data = cp.sum(log_likelihood) / m

    requires_grad = predicts.requires_grad
    if requires_grad:
        def CrossEntropy_backward(grad: cp.ndarray) -> cp.ndarray:
            p = softmax(predicts).data
            p[cp.arange(m), targets.data] -= 1
            p = p/m
            return p * grad
        depends_on = [Dependency(predicts, CrossEntropy_backward)]
    else:
        depends_on = []
    return (Tensor(data, requires_grad, depends_on))