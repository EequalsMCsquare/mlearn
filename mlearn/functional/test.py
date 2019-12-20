import c_func
import numpy as np

from time import time
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


view_shape = (1,24,24,8,5,5)
inputs = np.arange(1,3,28,28)
weights = np.random.randn(8,3,5,5)
bias = np.ones(8)

# 进行卷积操错
arr = np.lib.stride_tricks.as_strided(
            inputs, view_shape, inputs.strides*2).reshape(view_shape[1:])



dic = {}
dic['c_func'] = (c_func.sample_conv2d(arr,weights,bias))
dic['p_func'] = conv_2d(inputs,weights,bias)

