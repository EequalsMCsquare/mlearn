%module c_func

%{
  #define SWIG_FILE_WITH_INIT
  #include "c_func.h"
%}


%include "../../../numpy.i"
%include "cpointer.i"
%init %{
  import_array()
%}

%apply (double *IN_ARRAY5, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5){
  (double *inputs, int inputs_a, int inputs_b, int inputs_c, int inputs_d, int inputs_e),
(double *weights, int weights_a, int weights_b, int weights_c, int weights_d, int weights_e),
(double *bias, int bias_a, int bias_b, int bias_c, int bias_d, int bias_e)
};

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
    (double *inputs, int inputs_row, int inputs_col),
    (double *w, int w_row, int w_col),
    (double *b, int b_row, int b_col)};


%pointer_functions(double, doublep);

%include "c_func.h"
