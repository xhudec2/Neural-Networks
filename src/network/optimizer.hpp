#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../matrix/matrix.hpp"
#include "../matrix/printer.hpp"


// We implemented several optimizers but in the end Adam was the best, so we kept it.


struct Optimizer {
    virtual void step(Matrix& weights, Matrix& grad, Matrix& momentum,
                      Matrix& rmsprop) = 0;
};

struct Adam : Optimizer {
    DT learning_rate = 0.001;
    DT delta = 1e-8;
    DT beta1 = 0.9;
    DT beta2 = 0.999;
    DT weight_decay = 1e-2;
    size_t time = 1;

    Adam(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad, Matrix& momentum, Matrix& rmsprop) override;
};

struct RMSProp : Optimizer {
    DT learning_rate = 0.001;
    DT delta = 1e-8;
    DT rho = 0.9;

    RMSProp(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad, Matrix& momentum, Matrix& rmsprop) override;
};

struct SGD : Optimizer {
    DT learning_rate;
    DT alpha = 0.9;
    bool use_momentum = true;

    SGD(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad, Matrix& momentum,
              Matrix& rmsprop) override;
};

#endif