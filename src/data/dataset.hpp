#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>

#include "../matrix/matrix.hpp"
#include "../parser/csv.hpp"
#include "../constants.hpp"

struct Dataset {
    Matrix Xdata;
    Matrix ydata;
    size_t batch_size;
    size_t val_start;
    bool shuffle = false;

    Dataset(std::string Xpath, std::string ypath, size_t batch_size, size_t train_size)
        : batch_size{batch_size}, val_start{train_size} {
        shape_t Xshape{DATASET_SIZE, IMG_SIZE};
        shape_t yshape{DATASET_SIZE, 1};
        Xdata = Matrix(Xshape);
        ydata = Matrix(yshape);
        CSV::load(Xdata, Xpath);
        CSV::load(ydata, ypath);
    }

    void get_next_batch(size_t from, bool val, Matrix& Xbatch, Matrix& ybatch);
};

#endif
