#include "network.hpp"
#include <cassert>
#include <cstring>
#include "../matrix/printer.hpp"

void Linear::forward(const Matrix &input, Matrix &output) {
    // std::cout << "input:\n";
    // print(input);
    // std::cout << "weights:\n";
    // print(weights);

    assert(input.shape[1] == weights.shape[0]);
    assert(weights.shape[1] == output.shape[1] && input.shape[0] == output.shape[0]);

    memcpy(&inputs.data[0], &input.data[0], sizeof(DT) * input.size());
    mat_mul_mat(input, weights, output); // (batch, in_dim) x (in_dim, out_dim) = (batch, out_dim)
    output += bias;
    // std::cout << "mul output:\n";
    // print(output);
    sigma.diff(output, dSigma);     // (batch, 1, out_dim)
    sigma.apply(output);
    // std::cout << "mul output activation:\n";
    // print(output);
    // std::cout << "dSigma:\n";
    // print(dSigma);
    // std::cout << "\n\n";
}

void Linear::prepare_layer(size_t batch_size) {
    inputs  = Matrix({batch_size - batch_size + 1, shape[0]}); // in_dim x 1
    dSigma  = Matrix({shape[1], 1}); // 1 x out_dim
    dE_dOut = Matrix({shape[0], 1}); // 1 x out_dim
}

Matrix &Linear::backward(Matrix &dE_dy, bool last) {
    // std::cout << "grad:     " << gradient.shape << '\n';
    // std::cout << "bias_grad:" << bias_gradient.shape << '\n';
    // std::cout << "inputs:   " << inputs.shape << '\n';
    // std::cout << "dSigma:   " << dSigma.shape << '\n';
    // std::cout << "dE_dy:    " << dE_dy.shape << '\n';
    // std::cout << "dE_dOut:  " << dE_dOut.shape << '\n';
    // std::cout << '\n';
    bias_gradient *= 0;
    // print(dSigma.T());
    // print(dE_dy.T());
    // print(bias_gradient);
    bias_gradient += dSigma.T();
    // print(bias_gradient);
    bias_gradient *= dE_dy.T();  
    // print(bias_gradient);
    bias_gradient_accum += bias_gradient;

    gradient *= 0;
    // print(gradient);
    // print(inputs.T());
    gradient += inputs.T();
    // print(gradient);
    gradient *= dSigma.T();
    // print(gradient);
    gradient *= dE_dy.T();
    gradient_accum += gradient;
    // print(gradient);
    // print(gradient);
    // print(dE_dy);
    if (!last) {
        dSigma *= dE_dy;
        // print("dSigma *= dE_dy:");
        // print(dSigma);
        // print("weights:");
        // print(weights);
        // print(dSigma);
        mat_mul_mat(weights, dSigma, dE_dOut);
        // print(dE_dOut);
    }
    return dE_dOut;
}
