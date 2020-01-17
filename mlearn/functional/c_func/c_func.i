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
(double *weights, int weights_a, int weights_b, int weights_c, int weights_d, int weights_e)
};

%apply (double **ARGOUTVIEWM_ARRAY3, int *DIM1, int *DIM2, int *DIM3) {
    (double **arr_out, int *out_dim1, int *out_dim2, int *out_dim3)};

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
    (double *inputs, int inputs_row, int inputs_col),
    (double *w, int w_row, int w_col),
    (double *b, int b_row, int b_col)};


%pointer_functions(double, doublep);

%include "c_func.h"