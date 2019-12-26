#! /bin/sh

# # No OpenMP
# swig -python c_func.i&&\
# gcc -fPIC -fopenmp -m64 -Ofast -march=native -c c_func.c&&\
# gcc -fPIC -fopenmp -m64 -Ofast -march=native -c c_func_wrap.c -I/home/reco/.local/include/python -I/home/reco/.local/include/numpy/&&\
# gcc -shared -fopenmp -m64 -Ofast -march=native c_func.o  c_func_wrap.o -o _c_func.so&&\
# cp c_func_copy.py c_func.py&&\
# echo import c_func > test.py&&\
# python3 test.py


## With OpenMP
swig -python c_func.i&&\
gcc -fPIC -m64 -Ofast -march=native -fopenmp -c c_func.c&&\
gcc -fPIC -m64 -Ofast -march=native -fopenmp -c c_func_wrap.c -I/home/reco/.local/include/python -I/home/reco/.local/include/numpy/&&\
gcc -shared -m64 -Ofast -march=native -fopenmp c_func.o  c_func_wrap.o -o _c_func.so&&\
cp c_func_copy.py c_func.py&&\
echo import c_func > test.py&&\
python3 test.py


# # Only OpenMP and -Ofast
# swig -python c_func.i&&\
# gcc -fPIC -Ofast -fopenmp -c c_func.c&&\
# gcc -fPIC -Ofast -fopenmp -c c_func_wrap.c -I/home/reco/.local/include/python -I/home/reco/.local/include/numpy/&&\
# gcc -shared -Ofast -fopenmp c_func.o  c_func_wrap.o -o _c_func.so&&\
# cp c_func_copy.py c_func.py&&\
# echo import c_func > test.py&&\
# python3 test.py
