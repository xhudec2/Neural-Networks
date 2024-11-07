#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../matrix/matrix.hpp"

struct Optimizer {
    virtual void step(Matrix& weights, Matrix& grad) = 0;
};

struct SGD : Optimizer {
    DT learning_rate;

    SGD(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad) override {
        grad *= -learning_rate;
        weights += grad;
        grad = 0;
    }
};

#endif