#include "c_func.h"
#include <time.h>
// #include <omp.h>

// 卷积神经网络
double
dot_sum(double * t1, double * t2, int length)
{
    // 点乘
    // 这里不存在内存泄漏的问题
    double result = 0;

    for (int i = 0; i < length; i++)
        result += t1[i] * t2[i];
    return result;
}

double *
__sample_conv(double * strided_sample, int shapes[], double * weights, double * bias, int out_channels)
{
    // 传入一个stride后的tensor
    // e.g shape => (24,24,3,5,5)
    // 然后取后面三个维度的点乘之和并加以偏执值
    // out -> out_channel, height_out, width_out

    int _temp       = shapes[0] * shapes[1];
    int strides     = shapes[2] * shapes[3] * shapes[4];
    double * result = (double *) malloc(_temp * out_channels * sizeof(double));

    for (int k = 0; k < out_channels; k++) {
        for (int i = 0; i < _temp; i++)
            // 所有传入的数组都按一维度处理
            // 比如 24,24,3,5,5， 那么卷积框大小就是3,5,5
            // Result (8,24,24)
            result[k * _temp + i] = dot_sum(&strided_sample[i * strides],
                &weights[k * strides], strides)
              + bias[k];
    }
    return result;
}

double *
sample_conv2d(double * inputs, int inputs_a, int inputs_b, int inputs_c, int inputs_d, int inputs_e,
  double * weights, int weights_a, int weights_b, int weights_c, int weights_d, int weights_e,
  double * bias, int bias_a, int bias_b, int bias_c, int bias_d, int bias_e)
{
    int inputs_shape[] = { inputs_a, inputs_b, inputs_c, inputs_d, inputs_e };
    int out_channel    = weights_b;

    return __sample_conv(inputs, inputs_shape, weights, bias, out_channel);
}

// 全连接神经网络
double *
matmulAdd(double * inputs, int inputs_row, int inputs_col, double * w, int w_row, int w_col, double * b, int b_row,
  int b_col)
{
    // 接受一个转制后的权值， 也就是f_contiguous
    // 32,64
    // 64,10 => 10,64
    int i, j, k;
    double row_sum;
    int index       = 0;
    double * result = (double *) malloc(sizeof(double) * inputs_row * w_row);

    for (i = 0; i < inputs_row; i++) {
        for (k = 0; k < w_row; k++) {
            row_sum = 0;
            for (j = 0; j < inputs_col; j++)
                row_sum += inputs[i * inputs_col + j] * w[k * w_col + j];
            result[i * w_row + k] = row_sum + b[k];
        }
    }
    // terminated by signal SIGSEGV (Address boundary error)
    return result;
}

void
conv2d_test(int argc, char * argv[])
{
    // int coresNum = opm_get_num_procs();
    // printf("核心数 -> %d", coresNum);

    time_t start, end;
    // used for testing
    int length      = 24 * 24 * 3 * 5 * 5;
    double * inputs = (double *) malloc(length * sizeof(double));

    for (int i = 0; i < length; i++)
        inputs[i] = i;
    double * weights = (double *) malloc(8 * 3 * 25 * sizeof(double));
    for (int i = 0; i < 8 * 3 * 25; i++)
        weights[i] = i;

    double * bias = (double *) malloc(8 * sizeof(double));
    for (int i = 0; i < 8; i++)
        bias[i] = 5;

    int shape[]     = { 24, 24, 3, 5, 5 };
    double * result = (double *) malloc(24 * 24 * 8 * sizeof(double));
    int EPOCH;
    if (argc == 1)
        EPOCH = 5000;
    else
        EPOCH = atoi(argv[1]);

    printf("测试循环 %d次\n", EPOCH);
    // 卷积结果测试
    start = clock();
    for (int i = 0; i < EPOCH; i++)
        result = sample_conv2d(inputs, 24, 24, 3, 5, 5,
            weights, 1, 8, 3, 5, 5, bias, 8, 1, 1, 1, 1);
    end = clock();

    printf("Finished in %f Seconds\n", (double) (end - start) / CLOCKS_PER_SEC);

    // for (int i = 0; i < 8 * 24 * 24; i++)
    // {
    //   printf("%f[%d]", result[i], i);
    //   printf(((i % 5) != 4) ? "\t" : "\n");
    // }

    free(inputs);
    free(weights);
    free(bias);
    free(result);
} /* conv2d_test */

void
dense_test()
{
    double inputs[] = { 1, 2, 3, 4, 5, 6 };
    double w[]      = { 1, 2, 3, 4, 5, 6 };
    double b[]      = { 0, 0 };

    double * result = (double *) malloc(sizeof(double) * 4);

    result = matmulAdd(inputs, 2, 3, w, 2, 3, b, 1, 2);

    printf("\nmain: \n");
    for (int i = 0; i < 4; i++) {
        printf("%f  ", result[i]);
    }
}

int
main(int argc, char * argv[])
{
    dense_test();
    // conv2d_test(argc, argv);
}
