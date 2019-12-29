class ShapeError(Exception):

    def __init__(self, msg, shape_expect, shape_receive):
        """
            Tensor形状异常
            msg            -> Message     <str>
            shape_expect  -> 本来应该的形状 <Tuple>
            shape_receive -> 实际接受的形状 <Tuple>
        """
        _msg = msg + (f"\nShape expect -> {shape_expect}, but Shape receive -> {shape_receive}")
        super(ShapeError, self).__init__(_msg)

class LengthError(Exception):

    def __init__(self, msg, length_expect, length_receive):
        """
            长度异常
            msg            -> Message     <str>
            length_expect  -> 本来应该是多长 <int>
            length_receive -> 实际接受是多长 <int>
        """
        _msg = msg + (f"Length expect -> {length_expect}, but Length receive -> {length_receive}")
        super(LengthError, self).__init__(_msg)
