#ifndef HPARAMS_HPP
#define HPARAMS_HPP

#include "matrix/matrix.hpp"

struct Hparams {
    shape_t shape;
    float learning_rate;
    size_t num_epochs;
    size_t batch_size;
};

#endif
