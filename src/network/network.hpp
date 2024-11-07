#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include "linear.hpp"
#include "../matrix/matrix.hpp"
#include "../hparams.hpp"
#include "activation.hpp"
#include <cassert>


struct Network {
    std::vector<Linear> layers;
    ReLU relu;
    Sigmoid sigmoid;

    Network(shape_t shape) {
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            bool last_layer = i + 1 == shape.size() - 1;
            Activation &sigma = last_layer ? dynamic_cast<Activation &>(sigmoid) : dynamic_cast<Activation &>(relu);
            layers.emplace_back(shape[i], shape[i + 1], last_layer, sigma);
        }
        // initial weights to be the same as the torch version
        layers[0].weights.data = {0.1, -0.1, 0.2, 0.3, -0.3, 0.1};
        layers[1].weights.data = {0.1, -1, -0.2, 2, 0.3, -3};
    }

    void forward(const Matrix &, Matrix &);
    void backward(const Matrix &, const Matrix &);
    void update(float);
    void train(Hparams);
};


#endif
