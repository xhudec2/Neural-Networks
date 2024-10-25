#include "dataset.hpp"

#include <cstring>

void Dataset::get_next_batch(size_t from, bool val, Matrix &batch) {
    if (val) from += val_start;
    std::memcpy(batch.data.data(), &data.data.data()[from * IMG_SIZE],
                batch_size * IMG_SIZE);
}