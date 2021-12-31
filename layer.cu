#include "layer.h"

__global__
void conv2d(float *input, float *output, float *filters, float *bias, int width, int height, int filters_total, int shape_filters, int activation_name) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > width || y > height || z > filters_total) return;

    float conv = 0.0f;
    for (int i=0;i<shape_filters;i++) {
        for (int j=0;j<shape_filters;j++) {
            conv += input[(z * shape_filters * shape_filters) + (y + i) * width + (x + j)] * filters[(z * shape_filters * shape_filters) + (y + i) * width + (x + j)];
        }
    }
    conv += bias[z];
    void (*activation)(int) = activation_function(activation_name);
    float result = (*activation)(conv);
    output[z * shape_filters * shape_filters + y * shape_filters + x] = result;
    __syncthreads();
}

__global__
void averagePooling2d(float *input, float *output, int width, int height, int shape_filter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int new_width = ceil(width / shape_filter);
    int new_height = ceil(height / shape_filter);
    if (x > new_width || y > new_height) return;

    float result = 0.0f;
    for (int i=0;i<shape_filter;i++) {
        for (int j=0;j<shape_filter;j++) {
            result += input[(y + i) * width + (x + j)];
        }
    }
    output[y * new_width + x] = result / (shape_filter * shape_filter);
    __syncthreads();
}

__global__
void dense(float *input, float *output, float *weights, float *bias, int activation_name, int units) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > units || z > units) return;

    float result = input[x] * weights[x] + bias[x];
    void (*activation)(int) = activation_function(activation_name);
    output[x] = (*activation)(result);
    __syncthreads();
}
