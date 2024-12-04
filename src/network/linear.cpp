#include "linear.hpp"

#include <cassert>
#include <cstring>

#include "../matrix/printer.hpp"

void Linear::forward(const Matrix &input, Matrix &output, bool no_grad) {
    assert(input.shape[1] == weights.shape[0]);
    assert(weights.shape[1] == output.shape[1] &&
           input.shape[0] == output.shape[0]);

    if (!no_grad) {
        memcpy(&inputs.data[0], &input.data[0], sizeof(DT) * input.size());
    }
    mat_mul_mat(input, weights, output);
    output += bias;

    if (!no_grad) {
        sigma.diff(output, dSigma);
    }
    sigma.apply(output);
}

void Linear::prepare(size_t batch_size) {
    inputs = Matrix({batch_size, shape[0]});
    dSigma = Matrix({batch_size, shape[1]});
    dE_dOut = Matrix({batch_size, shape[0]});
}

void Linear::backward(Matrix &dE_dy, bool last) {
    for (size_t batch = 0; batch < dE_dy.shape[0]; batch++) {
        for (size_t j = 0; j < dE_dy.shape[1]; j++) {
            bias_grad.at(0, j) += dE_dy.at(batch, j) * dSigma.at(batch, j);
        }
    }

    for (size_t batch = 0; batch < dE_dy.shape[0]; batch++) {
        for (size_t i = 0; i < grad.shape[0]; i++) {
            for (size_t j = 0; j < grad.shape[1]; j++) {
                grad.at(i, j) += dE_dy.at(batch, j) * dSigma.at(batch, j) *
                                 inputs.at(batch, i);
            }
        }
    }

    if (!last) {
        dE_dOut = 0;
        for (size_t batch = 0; batch < dE_dy.shape[0]; batch++) {
            for (size_t i = 0; i < grad.shape[0]; i++) {
                for (size_t j = 0; j < grad.shape[1]; j++) {
                    dE_dOut.at(batch, i) += dE_dy.at(batch, j) *
                                            dSigma.at(batch, j) *
                                            weights.at(i, j);
                }
            }
        }
    }
}
