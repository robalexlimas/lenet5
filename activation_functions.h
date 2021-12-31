#include <cuda.h>
#include <math.h>

#ifndef ACTIVATIONS_FUNCTIONS
    #define ACTIVATIONS_FUNCTIONS
#endif

enum activations {
    RELU,
    TANH,
    SIGMOID
};

__device__ float relu(float a);
__device__ float tanh(float a);
__device__ float sigmoid(float a);

__device__ int activation_function(int activation);
