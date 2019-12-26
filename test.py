import numpy as np
from mlearn import functional as F
from time import time

start = time()
inputs = np.random.randn(1874, 32, 784)
weights = np.random.randn(784, 10)
bias = np.random.randn(1, 10)
gene = time() - start

start = time()
for batch in inputs:
    out = batch @ weights + bias
end = time() - start
print(out[0,:5])


start = time()
for batch in inputs:
    c_out = F.c_func.matmulAdd(batch, weights, bias)
c_end = time() - start
print(c_out[0,:5])

print("Generate Finished in-> %.5f"%gene)
print("Fortran Finished in -> %.5f"%end)
print("C Finished in -------> %.5f"%c_end)
