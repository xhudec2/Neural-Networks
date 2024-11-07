#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cassert>
#include <vector>

#include "../hparams.hpp"
#include "../matrix/matrix.hpp"
#include "activation.hpp"
#include "linear.hpp"

struct Network {
    std::vector<Linear> layers;
    ReLU relu;
    Identity identity;

    Network(shape_t shape) {
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            bool last_layer = i + 1 == shape.size() - 1;
            Activation &sigma = last_layer
                                    ? dynamic_cast<Activation &>(identity)
                                    : dynamic_cast<Activation &>(relu);
            layers.emplace_back(shape[i], shape[i + 1], last_layer, sigma);
        }

        // set initial weights to be the same as the torch version
        assert(layers.size() == 2);
        assert(layers[0].shape == shape_t({2, 3}));
        assert(layers[1].shape == shape_t({3, 2}));
        layers[0].weights.data = {0.1, -0.1, 0.2, 0.3, -0.3, 0.1};
        layers[1].weights.data = {0.1, -1, -0.2, 2, 0.3, -3};
    }

    void forward(const Matrix &, Matrix &);
    void backward(const Matrix &, const Matrix &);
    void update(float);
    void train(Hparams);
};

#endif
