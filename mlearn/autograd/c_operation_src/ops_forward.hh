#pragma once

#include <thread>
#include <stdlib.h>
#include <mutex>

// Add_forward
double *add_forward_1d(int a1, double* array1, int b1, double* array2);

double *add_forward_2d(int a1, int a2, double *array1, int b1, int b2, double *array2);

double *add_forward_3d(int a1, int a2, int a3, double *array1, int b1, int b2, int b3,
                  double *array2);

double *add_forward_4d(int a1, int a2, int a3, int a4, double *array1, int b1, int b2,
                  int b3, int b4, double *array2);

// Sub_forward

double *sub_forward_1d(int a1, double* array1, int b1, double* array2);

double *sub_forward_2d(int a1, int a2, double *array1, int b1, int b2, double *array2);

double *sub_forward_3d(int a1, int a2, int a3, double *array1, int b1, int b2, int b3,
                  double *array2);

double *sub_forward_4d(int a1, int a2, int a3, int a4, double *array1, int b1, int b2,
                  int b3, int b4, double *array2);
