#include <cuda.h>
#include <math.h>
#include "activation_functions.h"

#ifndef LAYER
    #define LAYER
#endif

__global__ void conv2d(float *input, float *output, float *filters, float *bias, int width, int height, int filters_total, int shape_filters, int activation_name)) ;
__global__ void averagePooling2d(float *input, float *output, int width, int height, int shape_filter=2);
__global__ void dense(float *input, float *output, float *weights, float *bias, int activation_name);
