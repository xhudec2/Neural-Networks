#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../matrix/matrix.hpp"
#include "../matrix/printer.hpp"

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

    void step(Matrix& weights, Matrix& grad, Matrix& momentum,
              Matrix& rmsprop) override {
        DT normalizing_beta1 = 1.0 - powf(beta1, time);
        DT normalizing_beta2 = 1.0 - powf(beta2, time);
        for (size_t i = 0; i < weights.size(); ++i) {
            momentum.data[i] =
                beta1 * momentum.data[i] + (1.0 - beta1) * grad.data[i];
            rmsprop.data[i] = beta2 * rmsprop.data[i] +
                              (1.0 - beta2) * grad.data[i] * grad.data[i];

            grad.data[i] =
                -(learning_rate * momentum.data[i] / normalizing_beta1) /
                (sqrt(rmsprop.data[i] / normalizing_beta2) + delta);

            weights.data[i] = weights.data[i] -
                              learning_rate * weight_decay * weights.data[i];
            weights.data[i] += grad.data[i];

            grad.data[i] = 0;
        }
        ++time;
    }
};

struct RMSProp : Optimizer {
    DT learning_rate = 0.001;
    DT delta = 1e-8;
    DT rho = 0.9;

    RMSProp(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad, Matrix& momentum,
              Matrix& rmsprop) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            rmsprop.data[i] = rho * rmsprop.data[i] +
                              (1.0 - rho) * grad.data[i] * grad.data[i];
            grad.data[i] =
                (-learning_rate / sqrt(rmsprop.data[i] + delta)) * grad.data[i];
            weights.data[i] += grad.data[i];
            grad.data[i] = 0;
        }
    }
};

struct SGD : Optimizer {
    DT learning_rate;
    DT alpha = 0.9;
    bool use_momentum = true;

    SGD(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad, Matrix& momentum,
              Matrix& rmsprop) override {
        if (use_momentum) {
            momentum *= alpha;
            momentum += grad;
        }
        grad *= -learning_rate;
        if (use_momentum) {
            grad += momentum;
        }

        weights += grad;

        grad = 0;
    }
};

#endif