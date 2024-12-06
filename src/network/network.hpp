#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cassert>
#include <vector>

#include "../matrix/matrix.hpp"
#include "activation.hpp"
#include "linear.hpp"
#include "optimizer.hpp"

struct Network {
    std::vector<Linear> layers;
    Optimizer& optimizer;
    ReLU relu;
    Identity identity;
    DT loss;               // Only used for easy access to print in main loop
    bool no_grad = false;  // When set to 'true', do not calculate gradients
    Matrix probs;
    Matrix dE_dy;

    Network(shape_t shape, Optimizer& optimizer) : optimizer{optimizer} {
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            bool last_layer = i + 1 == shape.size() - 1;
            Activation& sigma = last_layer ? dynamic_cast<Activation&>(identity)
                                           : dynamic_cast<Activation&>(relu);
            layers.emplace_back(shape[i], shape[i + 1], last_layer, sigma);
        }
    }

    void forward(const Matrix& inputs, Matrix& outputs, bool no_grad = false);
    void backward(const Matrix& outputs, const Matrix& targets);
    void update();
    void prepare(size_t batch_size);
    DT cross_entropy(const Matrix&, const Matrix&);
};

#endif
