#include "linear.hpp"
#include <cassert>
#include <cstring>
#include "../matrix/printer.hpp"

void Linear::forward(const Matrix &input, Matrix &output) {
    assert(input.shape[1] == weights.shape[0]);
    assert(weights.shape[1] == output.shape[1] && input.shape[0] == output.shape[0]);

    memcpy(&inputs.data[0], &input.data[0], sizeof(DT) * input.size());
    mat_mul_mat(input, weights, output);
    output += bias;
    sigma.diff(output, dSigma);
    sigma.apply(output);
}

void Linear::prepare_layer(size_t batch_size) {
    inputs  = Matrix({batch_size - batch_size + 1, shape[0]});
    dSigma  = Matrix({shape[1], 1});
    dE_dOut = Matrix({shape[0], 1});
}

void Linear::backward(Matrix &dE_dy, bool last) {
    bias_gradient *= 0;
    bias_gradient += dSigma.T();
    bias_gradient *= dE_dy.T();  
    bias_gradient_accum += bias_gradient;

    gradient *= 0;
    gradient += inputs.T();
    gradient *= dSigma.T();
    gradient *= dE_dy.T();
    gradient_accum += gradient;
    if (!last) {
        dSigma *= dE_dy;
        mat_mul_mat(weights, dSigma, dE_dOut);
    }
    dE_dy = dE_dOut;
}
