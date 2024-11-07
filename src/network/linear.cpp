#include "linear.hpp"

#include <cassert>
#include <cstring>

#include "../matrix/printer.hpp"

void Linear::forward(const Matrix &input, Matrix &output) {
    assert(input.shape[1] == weights.shape[0]);
    assert(weights.shape[1] == output.shape[1] &&
           input.shape[0] == output.shape[0]);

    memcpy(&inputs.data[0], &input.data[0], sizeof(DT) * input.size());
    mat_mul_mat(input, weights, output);
    output += bias;

    sigma.diff(output, dSigma);
    sigma.apply(output);
}

void Linear::prepare_layer(size_t batch_size) {
    inputs = Matrix({batch_size, shape[0]});
    dSigma = Matrix({batch_size, shape[1]});
    dE_dOut = Matrix({batch_size, shape[0]});
}

void Linear::backward(Matrix &dE_dy, bool last) {
    for (size_t batch = 0; batch < dE_dy.shape[0]; batch++) {
        for (size_t j = 0; j < dE_dy.shape[1]; j++) {
            bias_grad[{0, j}] += dE_dy[{batch, j}] * dSigma[{batch, j}];
        }
    }

    for (size_t batch = 0; batch < dE_dy.shape[0]; batch++) {
        for (size_t i = 0; i < grad.shape[0]; i++) {
            for (size_t j = 0; j < grad.shape[1]; j++) {
                grad[{i, j}] +=
                    dE_dy[{batch, j}] * dSigma[{batch, j}] * inputs[{batch, i}];
            }
        }
    }

    if (!last) {
        dE_dOut = 0;
        for (size_t batch = 0; batch < dE_dy.shape[0]; batch++) {
            for (size_t i = 0; i < grad.shape[0]; i++) {
                for (size_t j = 0; j < grad.shape[1]; j++) {
                    dE_dOut[{batch, i}] += dE_dy[{batch, j}] *
                                           dSigma[{batch, j}] * weights[{i, j}];
                }
            }
        }
    }

    dE_dy = dE_dOut;
}
