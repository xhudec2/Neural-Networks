#include "optimizer.hpp"
#include <cmath>

void Adam::step(Matrix& weights, Matrix& grad, Matrix& momentum,
          Matrix& rmsprop) {
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

void RMSProp::step(Matrix& weights, Matrix& grad, Matrix& momentum,
          Matrix& rmsprop) {
    for (size_t i = 0; i < weights.size(); ++i) {
        rmsprop.data[i] = rho * rmsprop.data[i] +
                          (1.0 - rho) * grad.data[i] * grad.data[i];
        grad.data[i] =
            (-learning_rate / sqrt(rmsprop.data[i] + delta)) * grad.data[i];
        weights.data[i] += grad.data[i];
        grad.data[i] = 0;
    }
}

void SGD::step(Matrix& weights, Matrix& grad, Matrix& momentum,
          Matrix& rmsprop) {
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

