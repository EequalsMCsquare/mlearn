#! /bin/sh
swig -python c_func.i&&\
gcc -fPIC -Ofast -mavx2 -mfpmath=sse -c c_func.c&&\
gcc -fPIC -Ofast -mavx2 -mfpmath=sse -c c_func_wrap.c -I/home/reco/.local/include/python -I/home/reco/.local/include/numpy/&&\
gcc -shared -Ofast -mavx2 -mfpmath=sse c_func.o  c_func_wrap.o -o _c_func.so&&\
cp c_func_copy.py c_func.py&&\
python3 test.py
