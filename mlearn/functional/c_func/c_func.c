#include "c_func.h"

double dot_sum(double *t1, double *t2, int length)
{
  // 点乘
  double result = 0;
  for (int i = 0; i < length; i++)
    result += t1[i] * t2[i];
  return result;
}

double *__sample_conv(double *strided_sample, int shapes[], double *weights, double *bias, int out_channels)
{
  // 传入一个stride后的tensor
  // e.g shape => (24,24,3,5,5)
  // 然后取后面三个维度的点乘之和并加以偏执值
  // out -> out_channel, height_out, width_out

  int _temp = shapes[0] * shapes[1];             
  int strides = shapes[2] * shapes[3] * shapes[4];
  double *result = (double *)malloc(_temp * out_channels * sizeof(double)); 

  // Used for Debug=====
  // for(int j = 0; j < _temp; j++){
  //   printf("[%d] %f\n",j,strided_sample[j * strides]);
  // }
  // ===================

  for (int k = 0; k < out_channels; k++) // 
  {
    for (int i = 0; i < _temp; i++)
      // Check 
      // 所有传入的数组都按一维度处理
      // 比如 24,24,3,5,5， 那么卷积框大小就是3,5,5

      // Result (8,24,24)
      result[k * _temp + i] = dot_sum(&strided_sample[i * strides],
                                      &weights[k * strides], strides) +
                              bias[k]; 
  }
  return result;
}

double *sample_conv2d(double *inputs, int inputs_a, int inputs_b, int inputs_c, int inputs_d, int inputs_e,
                      double *weights, int weights_a, int weights_b, int weights_c, int weights_d, int weights_e,
                      double *bias, int bias_a, int bias_b, int bias_c, int bias_d, int bias_e)
{
  int inputs_shape[] = {inputs_a, inputs_b, inputs_c, inputs_d, inputs_e};
  int out_channel = weights_b;
  double *result = (double *)malloc(sizeof(double) * out_channel * inputs_a * inputs_b);
  
  return  __sample_conv(inputs, inputs_shape, weights, bias, out_channel);
}



int main()
{
  // used for testing
  int length = 24 * 24 * 3 * 5 * 5;
  double *inputs = (double *)malloc(length * sizeof(double));
  for (int i = 0; i < length; i++)
    inputs[i] = i;
  double *weights = (double *)malloc(8 * 3 * 25 * sizeof(double));
  for (int i = 0; i < 8 * 3 * 25; i++)
    weights[i] = i;

  double *bias = (double *)malloc(8 * sizeof(double));
  for (int i = 0; i < 8; i++)
    bias[i] = i;


  int shape[] = {24, 24, 3, 5, 5};
  double *result = (double *)malloc(24 * 24 * 8 * sizeof(double));

  double dot_sumR;
  // 卷积结果测试
  result = sample_conv2d(inputs, 24, 24, 3, 5, 5,
                         weights, 1, 8, 3, 5, 5, bias, 8, 1, 1, 1, 1);


  for (int i = 0; i < 8 * 24 * 24; i++)
  {
    printf("%f[%d]", result[i], i);
    printf(((i % 5) != 4) ? "\t" : "\n");
  }
  // 结果比python少三个数 
}