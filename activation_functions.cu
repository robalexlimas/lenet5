#include "activation_functions.h"

__device__
float relu(float a) {
    return a < 0.0f ? 0.0f : a;
}

__device__
float tanh(float a) {
    return (2 / (1 + exp(-2 * a))) - 1;
}

__device__
float sigmoid(float a) {
    return 1 / (1 + exp(-a));
}

__device__
int activation_function(int activation) {
    int function = 0;
    switch(activation) {
        case RELU:
            function = &relu;
            break;
        case TANH:
            function = &tanh;
            break;
        case SIGMOID:
            function = &sigmoid;
            break;
    }
    return function;
}
