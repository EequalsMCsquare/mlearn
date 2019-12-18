#include "ops_forward.hh"

// Add
void __add(double *out, double *t1, int t1_shape[], int a_dim, double *t2,
           int t2_shape[], int b_dim) {
  int arr_length = 1;
  for (int i = 0; i < a_dim; i++)
    arr_length *= t1_shape[i];
  double *result = (double *)malloc(arr_length * sizeof(double));
  for (int i = 0; i < arr_length; i++)
    out[i] = t1[i] + t2[i];
}

double *add_forward_1d(int a1, double *array1, int b1, double *array2) {
  int t1_shape[] = {a1};
  int t2_shape[] = {b1};
  double *result = (double *)malloc(sizeof(double) * a1);
  __add(result, array1, t1_shape, 1, array2, t2_shape, 1);
  return result;
}



double *add_forward_2d(int a1, int a2, double *array1, int b1, int b2,
                       double *array2) {
  int t1_shape[] = {a1, a2};
  int t2_shape[] = {b1, b2};
  double *result = (double *)malloc(sizeof(double) * a1 * a2);
  __add(result, array1, t1_shape, 2, array2, t2_shape, 2);
  return result;
}

double *add_forward_3d(int a1, int a2, int a3, double *array1, int b1, int b2,
                       int b3, double *array2) {
  int t1_shape[] = {a1, a2, a3};
  int t2_shape[] = {b1, b2, b3};
  double *reuslt = (double *)malloc(a1 * a2 * a3 * sizeof(double));

  __add(reuslt, array1, t1_shape, 3, array2, t2_shape, 3);
  return reuslt;
}

double *add_forward_4d(int a1, int a2, int a3, int a4, double *array1, int b1,
                       int b2, int b3, int b4, double *array2) {
  int t1_shape[] = {a1, a2, a3, a4};
  int t2_shape[] = {b1, b2, b3, b4};

  double *reuslt = (double *)malloc(a1 * a2 * a3 * a4 * sizeof(double));

  __add(reuslt, array1, t1_shape, 4, array2, t2_shape, 4);
  return reuslt;
}

// neg
double *__neg(double *t, int t_shape[], int a_dim) {
  int arr_length = 1;
  for (int i = 0; i < a_dim; i++)
    arr_length *= t_shape[i];
  double *result = (double *)malloc(arr_length * sizeof(double));
  for (int i = 0; i < arr_length; i++)
    result[i] = t[i] * -1;
  return result;
}
// Sub
double *__sub(double *t1, int t1_shape[], int a_dim, double *t2, int t2_shape[],
              int b_dim) {
  int arr_length = 1;
  for (int i = 0; i < a_dim; i++)
    arr_length *= t1_shape[i];
  double *result = (double *)malloc(arr_length * sizeof(double));
  for (int i = 0; i < arr_length; i++)
    result[i] = t1[i] - t2[i];
  return result;
}

double *sub_forward_1d(int a1, double *array1, int b1, double *array2) {
  int t1_shape[] = {a1};
  int t2_shape[] = {b1};

  return __sub(array1, t1_shape, 1, array2, t2_shape, 1);
}

double *sub_forward_2d(int a1, int a2, double *array1, int b1, int b2,
                       double *array2) {
  int t1_shape[] = {a1, a2};
  int t2_shape[] = {b1, b2};

  return __sub(array1, t1_shape, 2, array2, t2_shape, 2);
}

double *sub_forward_3d(int a1, int a2, int a3, double *array1, int b1, int b2,
                       int b3, double *array2) {
  int t1_shape[] = {a1, a2, a3};
  int t2_shape[] = {b1, b2, b3};

  return __sub(array1, t1_shape, 3, array2, t2_shape, 3);
}

double *sub_forward_4d(int a1, int a2, int a3, int a4, double *array1, int b1,
                       int b2, int b3, int b4, double *array2) {
  int t1_shape[] = {a1, a2, a3, a4};
  int t2_shape[] = {b1, b2, b3, b4};

  return __sub(array1, t1_shape, 4, array2, t2_shape, 4);
}