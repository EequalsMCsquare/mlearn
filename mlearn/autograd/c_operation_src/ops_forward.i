%module ops_forward

%{
  #define SWIG_FILE_WITH_INIT
  #include "ops_forward.hh"
%}

%include "numpy.i"
%init %{
  import_array()
%}


%apply (int DIM1, double* IN_ARRAY1){(int a1, double* array1), (int b1, double* array2)}
%apply (int DIM1, int DIM2, double* IN_ARRAY2){(int a1, int a2, double* array1), (int b1, int b2, double* array2)}
%apply (int DIM1, int DIM2,int DIM3, double* IN_ARRAY3){(int a1, int a2,int a3,double* array1), (int b1, int b2, int b3,double* array2)}
%apply (int DIM1, int DIM2,int DIM3, int DIM4, double* IN_ARRAY4){(int a1, int a2,int a3,int a4,double* array1), (int b1, int b2, int b3,int b4,double* array2)}



%include "ops_forward.hh"