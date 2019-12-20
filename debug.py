from mlearn.functional import c_func
import numpy as np
import mlearn.functional as F
from time import time
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

view_shape = (1,24,24,3,5,5)
inputs = np.arange(3*28*28).reshape(3,28,28)
weights = np.random.randn(8,3,5,5)
bias = np.ones(8)

# 进行卷积操错
arr = np.lib.stride_tricks.as_strided(
            inputs, view_shape, inputs.strides*2).reshape(view_shape[1:])



dic = {}
dic['c_func'] = np.array([c_func.sample_conv2d(arr,weights.reshape(1,8,3,5,5),bias.reshape(-1,1,1,1,1))])
dic['p_func'] = F.conv_2d(inputs.reshape(1,3,28,28),weights,bias)
dic['pc_func'] = F.c_conv_2d(inputs.reshape(1,3,28,28),weights,bias)

a = dic['c_func']
b = dic['p_func']
c = dic['pc_func']