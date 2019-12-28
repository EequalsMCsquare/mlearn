
import numpy as np
from .tensor import Tensor

class Parameter(Tensor):
    def __init__(self, *shape, bias=False, init=1) -> None:
        """
        參數初始化
        shape -> 參數的形狀  <Tuple>
        bias  -> 是否是bias <bool>

        可用的initializer:

        """
        data = np.random.randn(*shape) * init
        super().__init__(data, requires_grad=True)
    def __repr__(self):
        return f"可訓練的參數\n{super().__repr__()}"

class param_init:
    r"""
        Reference:
            - https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
    """
    @staticmethod
    def xavier_init(in_num:int, out_num:int):
        r"""
            in_num  -> Weights_in  個數
            out_num -> Weights_out 個數

            >>> Parameter * xavier_init(in_num, out_num)
        """
        if not isinstance(in_num, int):
            raise ValueError(f"in_num必須是一個整數， 而不是 {type(in_num)}")
        if not isinstance(out_num, int):
            raise ValueError(f"out_num必須是一個整數， 而不是 {type(in_num)}")

        return np.sqrt(2./(in_num + out_num))

    @staticmethod
    def He_init(in_num:int):
        r"""
            in_num -> Weights_in 個數

            >>> Parameter * He_init(in_num)
        """
        if not isinstance(in_num, int):
            raise ValueError(f"in_num必須是一個整數， 而不是 {type(in_num)}")
        return np.sqrt(1./in_num)
