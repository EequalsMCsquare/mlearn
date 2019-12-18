import ops_forward
import numpy as np
from ctypes import c_double
from time import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


print("\tADD")
print("\tONE DIM ARRAY")
length = 500000
start = time()
a = np.random.randn(length)
b = np.random.randn(length)


print("Used %.5f sec to Generate Array"%(time() -start))

start = time()
out = ops_forward.add_forward_1d(a,b)
p = (c_double * length).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))

start = time()
out = a + b
print("Numpy Finished in %.5f sec"%(time() - start))

print()
print("\tTWO DIM ARRAY")
row = 20000
col = 10000
start = time()
a = np.random.randn(row , col)
b = np.random.randn(row, col)
print("Used %.5f sec to Generate Array"%(time() -start))
start = time()
out = ops_forward.add_forward_2d(a,b) 
p = (c_double * row * col).from_address(int(out))
out = np.ctypeslib.as_array(p).reshape(row,col)
print("C Finished in %.5f sec"%(time() - start))
start = time()
out = a + b
print("Numpy Finished in %.5f sec"%(time() - start))

del a,b


print()
print("THREE DIM ARRAY")
_a = 2000
_b = 1000
_c = 100
start = time()
a = np.random.randn(_a,_b,_c)
b = np.random.randn(_a,_b,_c)
print("Used %.5f sec to Generate Array"%(time() -start))
start = time()
out = ops_forward.add_forward_3d(a,b)
p = (c_double * _a * _b * _c).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))
start = time()
a + b
print("Numpy Finished in %.5f sec"%(time() - start))

del a,b


print()
print("FOUR DIM ARRAY")
_a = 500
_b = 100
_c = 100
_d = 50
start = time()
a = np.random.randn(_a,_b,_c,_d)
b = np.random.randn(_a,_b,_c,_d)
print("Used %.5f sec to Generate Array"%(time() -start))
start = time()
out = ops_forward.add_forward_4d(a,b)
p = (c_double * _a * _b * _c * _d).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))
start = time()
a + b
print("Numpy Finished in %.5f sec"%(time() - start))

del a,b

"""
    SUB
"""


print("\n\t SUB")
print()
print("\tONE DIM ARRAY")
length = 500000
start = time()
a = np.random.randn(length)
b = np.random.randn(length)
print("Used %.5f sec to Generate Array"%(time() -start))

start = time()
out = ops_forward.sub_forward_1d(a,b)
p = (c_double * length).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))

start = time()
out = a + b
print("Numpy Finished in %.5f sec"%(time() - start))
print()

del a,b


print()
print("\tTWO DIM ARRAY")
row = 20000
col = 10000
start = time()
a = np.random.randn(row , col)
b = np.random.randn(row, col)
print("Used %.5f sec to Generate Array"%(time() -start))
start = time()
out = ops_forward.sub_forward_2d(a,b)
p = (c_double * row * col).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))
start = time()
a + b
print("Numpy Finished in %.5f sec"%(time() - start))
del a,b

print()
print("THREE DIM ARRAY")
_a = 2000
_b = 1000
_c = 100
start = time()
a = np.random.randn(_a,_b,_c)
b = np.random.randn(_a,_b,_c)
print("Used %.5f sec to Generate Array"%(time() -start))
start = time()
out = ops_forward.sub_forward_3d(a,b)
p = (c_double * _a * _b * _c).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))
start = time()
a + b
print("Numpy Finished in %.5f sec"%(time() - start))
del a,b

print()
print("FOUR DIM ARRAY")
_a = 500
_b = 100
_c = 100
_d = 50
start = time()
a = np.random.randn(_a,_b,_c,_d)
b = np.random.randn(_a,_b,_c,_d)
print("Used %.5f sec to Generate Array"%(time() -start))
start = time()
out = ops_forward.sub_forward_4d(a,b)
p = (c_double * _a * _b * _c * _d).from_address(int(out))
out = np.ctypeslib.as_array(p)
print("C Finished in %.5f sec"%(time() - start))
start = time()
a + b
print("Numpy Finished in %.5f sec"%(time() - start))

del a,b

