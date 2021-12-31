#include <cuda.h>
#include <math.h>
#include "activation_functions.h"

#ifndef LAYER
    #define LAYER
#endif

__global__ void conv2d(float *input, float *output, float *filters, int widht, int height, int filters, int shape_filters, int activation_function);
__global__ void averagePooling2d(float *input, float *output, int shape_filter=2);
__global__ void dense(float *input, float *output, int activation_function);
