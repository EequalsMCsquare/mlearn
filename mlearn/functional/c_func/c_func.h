#include <stdio.h>
#include <stdlib.h>

double *sample_conv2d(double *inputs, int inputs_a, int inputs_b, int inputs_c,
                      int inputs_d, int inputs_e, double *weights,
                      int weights_a, int weights_b, int weights_c,
                      int weights_d, int weights_e, double *bias, int bias_a,
                      int bias_b, int bias_c, int bias_d, int bias_e);

double *matmulAdd(double *inputs, int inputs_row, int inputs_col, double *w,
                  int w_row, int w_col, double *b, int b_row, int b_col);
