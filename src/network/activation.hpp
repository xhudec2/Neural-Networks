#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>

#include "../matrix/matrix.hpp"

struct Activation {
    virtual void apply(Matrix &) = 0;
    virtual void diff(const Matrix &, Matrix &) = 0;
};

struct ReLU : Activation {
    virtual void apply(Matrix &output) override {
        for (size_t i = 0; i < output.size(); ++i) {
            output.data[i] = output.data[i] > 0 ? output.data[i] : 0;
        }
    }
    virtual void diff(const Matrix &output, Matrix &dSigma) override {
        for (size_t i = 0; i < output.size(); ++i) {
            dSigma.data[i] = output.data[i] > 0 ? 1 : 0;
        }
    }
};

struct Sigmoid : Activation {
    virtual void apply(Matrix &output) override {
        assert(output.size() == 1);
        output.data[0] = 1 / (1 + std::exp(-output.data[0]));
    }
    virtual void diff(const Matrix &output, Matrix &dSigma) override {
        assert(output.size() == 1);
        DT val = 1 / (1 + std::exp(-output.data[0]));
        dSigma.data[0] = val * (1 - val);
    }
};

struct Identity : Activation {
    virtual void apply(Matrix &output) override { (void)output; }
    virtual void diff(const Matrix &output, Matrix &dSigma) override {
        (void)output;
        dSigma = 1;
    }
};

#endif
