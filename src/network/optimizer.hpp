#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../matrix/matrix.hpp"
#include "../matrix/printer.hpp"

struct Optimizer {
    virtual void step(Matrix& weights, Matrix& grad) = 0;
    virtual void step(Matrix& weights, Matrix& grad, Matrix &momentum) = 0;
};

struct SGD : Optimizer {
    DT learning_rate;
    float alpha = 0.9;

    SGD(DT learning_rate) : learning_rate{learning_rate} {};

    void step(Matrix& weights, Matrix& grad) override {
        grad *= -learning_rate;
        weights += grad;
        grad = 0;
    }
    
    void step(Matrix& weights, Matrix& grad, Matrix &momentum) override {
        bool quit = false;
        for (DT val : momentum.data) {
            if (isnan(val)) {
                print("momentum is nan");
                quit = true;
            }
        }
        for (DT val : grad.data) {
            if (isnan(val)) {
                print("grad is nan");
                quit = true;
            }
        }
        momentum *= alpha;
        grad += momentum;
        momentum = grad;
        grad *= -learning_rate;

        for (DT val : weights.data) {
            if (isnan(val)) {
                print("weights is nan");
                quit = true;
            }
        }
        weights += grad;

        for (DT val : weights.data) {
            if (isnan(val)) {
                print("weights is nan after computation");
                quit = true;
            }
        }
        
        for (DT val : momentum.data) {
            if (isnan(val)) {
                print("momentum is nan after computation");
                quit = true;
            }
        }
        for (DT val : grad.data) {
            if (isnan(val)) {
                print("grad is nan after computation");
                quit = true;
            }
        }
        if (quit) exit(1);
    
        grad = 0;
    }
};

#endif