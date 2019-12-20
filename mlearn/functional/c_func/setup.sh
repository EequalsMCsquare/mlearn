#! /bin/sh
swig -python c_func.i&&\
gcc -fPIC -O3 -c c_func.c&&\
gcc -fPIC -O3 -c c_func_wrap.c -I/usr/local/include/python3.7 -I/usr/local/include/numpy&&\
gcc -shared c_func.o -O3 c_func_wrap.o -o _c_func.so&&\

python3 test.py
