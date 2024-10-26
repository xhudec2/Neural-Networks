#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include "../matrix/matrix.hpp"
#include <cmath>
#include <cassert>

struct Hparams {};


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


struct Linear {
    shape_t shape;
    Matrix weights; // in_dim x out_dim
    Matrix bias;
    Matrix gradient; // in_dim x out_dim
    Matrix bias_gradient;
    bool output;
    Activation &sigma;

    // Matrix batch_gradient; // batch x in_dim x out_dim
    Matrix inputs; // ~~batch x~~ in_dim x 1
    Matrix dSigma; // ~~batch x~~ 1 x out_dim
    Matrix dE_dOut; // ~~batch x~~ 1 x out_dim - passed to layer below

    Linear(size_t in_dim, size_t out_dim, bool output, Activation &sigma)
        : shape{{in_dim, out_dim}}, 
          weights{random_normal(shape, 0, 2.0 / (shape[1] + product(shape)))},
          bias{random_normal({1, shape[1]}, 0, 2.0 / (shape[1] + product(shape)))},
            //   weights{shape, .1}, bias{{1, shape[1]}, .1},
          gradient{shape}, bias_gradient{{1, shape[1]}},
          output{output}, sigma{sigma} {};

    void prepare_layer(size_t);

    void forward(const Matrix &, Matrix &);
    Matrix &backward(Matrix &, bool); // can be changed to look line Network::backward
};

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
    }

    void forward(const Matrix &, Matrix &);
    void backward(const Matrix &, const Matrix &);
    void update();
    void train();
};


#endif
