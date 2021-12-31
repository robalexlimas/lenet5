#include "layer.h"

__global__
void conv2d(float *input, float *output, float *filters, int widht, int height, int filters, int shape_filters, int activation_name) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    void (*activation)(int) = activation_function(activation_name);
    float result = (*activation)(0.0f);
}

__global__
void averagePooling2d(float *input, float *output, int shape_filter) {

}

__global__
void dense(float *input, float *output, int activation_name) {
    void (*activation)(int) = activation_function(activation_name);
    float result = (*activation)(0.0f);
}
