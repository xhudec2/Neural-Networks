#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "../matrix/matrix.hpp"
#include "activation.hpp"

struct Linear {
    shape_t shape;
    Matrix weights; // in_dim x out_dim
    Matrix bias;
    Matrix gradient; // in_dim x out_dim
    Matrix bias_gradient;
    Matrix gradient_accum; // in_dim x out_dim
    Matrix bias_gradient_accum;
    bool output;
    Activation &sigma;

    Matrix inputs; // ~~batch x~~ in_dim x 1
    Matrix dSigma; // ~~batch x~~ 1 x out_dim
    Matrix dE_dOut; // ~~batch x~~ 1 x out_dim - passed to layer below

    Linear(size_t in_dim, size_t out_dim, bool output, Activation &sigma)
        : shape{{in_dim, out_dim}}, 
          weights{random_normal(shape, 0, std::sqrt(2.0 / shape[0]))}, // He init
          bias{zeros({1, shape[1]})},
          gradient{shape}, bias_gradient{{1, shape[1]}},
          gradient_accum{shape}, bias_gradient_accum{{1, shape[1]}},
          output{output}, sigma{sigma} {};

    void prepare_layer(size_t);
    void forward(const Matrix &, Matrix &);
    void backward(Matrix &, bool);
};

#endif
