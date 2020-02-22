
#include <cstddef>
#include <omp.h>
#include <iostream>

extern "C" 
{
    void dot_sum(double *result, double *x, double *w, std::size_t *length);   
}

// inline void dotSum(double &result, double *t1, double *t2, std::size_t &length){
//     result = 0;
//     std::size_t i;
//     // TODO:  这里用smid重写
//     for(i = 0; i < length; i++)
//         result += t1[i] * t2[i];
// }

inline void sampleConv2d(double *sample_result, 
double *x, 
double *w,
std::size_t *shapes,
std::size_t &out_channels){
    // x shape => (24, 24, 3, 5, 5)
    // w shape => (16, 3, 5, 5)
    // b shape => (16)
    // sample_result已经被malloc过

    std::size_t _temp_ = shapes[0] * shapes[1]; 
    std::size_t _strides_ = shapes[2] * shapes[3] * shapes[4];
    std::size_t idx_1, idx_2;
    for (std::size_t k = 0; k < out_channels; k++){
        idx_1 = k * _temp_;
        idx_2 = k * _strides_;
        for(std::size_t i = 0; i < _temp_; i++)
        dot_sum(&sample_result[idx_1 + i], &x[i * _strides_], 
        &w[idx_2], &_strides_);
        }
}

void batchConv2d(double *batch_result, 
double *x,
double *w,
std::size_t *shapes,
std::size_t &out_channels){
    // x_shape => (32,24,24,3,5,5)
    // result_shape => (32,16,24,24)
    // batch_result已经被malloc了
    const unsigned int x_strides = shapes[1] * shapes[2] * shapes[3] * shapes[4] * shapes[5];
    const unsigned int result_strides = out_channels * shapes[1] * shapes[2];
    
    #pragma omp parallel for num_threads(omp_get_num_procs())
    for (std::size_t i = 0; i < shapes[0]; i++)
        sampleConv2d(&batch_result[i * result_strides], 
            &x[i * x_strides], w, &shapes[1], out_channels);
    
}