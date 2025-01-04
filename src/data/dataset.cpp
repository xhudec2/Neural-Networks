#include "dataset.hpp"
#include <cstring>
#include <algorithm>


void Dataset::get_next_batch(size_t from, bool validation, Matrix &Xbatch, Matrix &ybatch) {
    if (validation) from += val_start;
    std::memcpy(Xbatch.data.data(), &Xdata.data.data()[from * IMG_SIZE],
                batch_size * IMG_SIZE * sizeof(DT));
    std::memcpy(ybatch.data.data(), &ydata.data.data()[from * 1],
                batch_size * 1 * sizeof(DT));
}

// similar to the reference algorithm https://en.cppreference.com/w/cpp/algorithm/random_shuffle
// we at first shuffle the entire train set, then only the part used for actual
// training and leave validation as is.
void Dataset::shuffle(bool shuffle_all) {
    size_t shuffle_size = TRAIN_SIZE;
    if (shuffle_all)
        shuffle_size = DATASET_SIZE;

    std::vector<size_t> indices(shuffle_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    for (size_t i = 0; i < shuffle_size; ++i) {
        std::swap(ydata.data[i], ydata.data[indices[i]]);
        std::swap_ranges(
            &Xdata.data.data()[i * IMG_SIZE],
            &Xdata.data.data()[(i + 1) * IMG_SIZE],
            &Xdata.data.data()[indices[i] * IMG_SIZE]);
    }
}
