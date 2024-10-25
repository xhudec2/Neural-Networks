#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <utility>

#include "../matrix/matrix.hpp"
#include "../parser/csv_reader.hpp"

const size_t IMG_SIZE = 28 * 28;
const size_t DATASET_SIZE = 60000;

struct Dataset {
    Matrix data;
    size_t batch_size;
    size_t val_start;
    bool shuffle = false;

    Dataset(std::string path, size_t batch_size, size_t train_size)
        : batch_size{batch_size}, val_start{train_size} {
        std::vector<size_t> shape{DATASET_SIZE, IMG_SIZE};
        CSV file(shape, path);
        data = std::move(file.data);
    }

    void get_next_batch(size_t, bool, Matrix &);
};

#endif
