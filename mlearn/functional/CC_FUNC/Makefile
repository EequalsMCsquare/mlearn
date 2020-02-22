
INC := $(shell pwd)/../../../include


CC_FLAGS = -Ofast -fopenmp -mavx -ffast-math -Wall -Wextra -Wno-unused-parameter

INCLUDE = -I$(INC)/python -I$(INC)/numpy

py_conv: f_dotSum
	g++ -fPIC -shared -I$(INC)/python src/py_conv.cxx build/fortran_F.o -o py_conv.so $(CC_FLAGS)

f_dotSum: src/fortran_F.f95
	gcc -c src/fortran_F.f95 -lgfortran -o build/fortran_F.o -Ofast -ffast-math -mavx2

clean: 
	rm build/*.o