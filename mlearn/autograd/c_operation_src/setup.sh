#! /bin/sh
swig -c++ -python ops_forward.i&&\
g++ -fPIC -c ops_forward.cpp &&\
g++ -fPIC -c ops_forward_wrap.cxx -I/usr/local/include/python3.7 -I/usr/local/include/numpy&&\
g++ -shared ops_forward.o ops_forward_wrap.o -o _ops_forward.so&&\
python3 test.py
