#include "dataset.hpp"

#include <cstring>

void Dataset::get_next_batch(size_t from, bool val, Matrix &Xbatch, Matrix &ybatch) {
    if (val) from += val_start;
    std::memcpy(Xbatch.data.data(), &Xdata.data.data()[from * IMG_SIZE],
                batch_size * IMG_SIZE * sizeof(DT));
    std::memcpy(ybatch.data.data(), &ydata.data.data()[from * 1],
                batch_size * 1 * sizeof(DT));
}