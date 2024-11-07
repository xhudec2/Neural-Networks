#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cassert>
#include <vector>

#include "../hparams.hpp"
#include "../matrix/matrix.hpp"
#include "activation.hpp"
#include "linear.hpp"
#include "optimizer.hpp"

struct Network {
    std::vector<Linear> layers;
    Optimizer& optimizer;
    ReLU relu;
    Identity identity;

    Network(shape_t shape, Optimizer& optimizer) : optimizer{optimizer} {
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            bool last_layer = i + 1 == shape.size() - 1;
            Activation &sigma = last_layer
                                    ? dynamic_cast<Activation &>(identity)
                                    : dynamic_cast<Activation &>(relu);
            layers.emplace_back(shape[i], shape[i + 1], last_layer, sigma);
        }
    }

    void forward(const Matrix &, Matrix &);
    void backward(const Matrix &, const Matrix &);
    void update();
    void train(Hparams);
};

#endif
