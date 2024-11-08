#include "helpers.hpp"

#include <math.h>

// logits - batch_size x fan_out
// probs  - batch_size x fan_out
void softmax(const Matrix& logits, Matrix& probs) {
    for (size_t batch = 0; batch < logits.shape[0]; batch++) {
        DT exp_sum = 0;
        for (size_t j = 0; j < logits.shape[1]; j++) {
            probs.at(batch, j) = expf(logits.at(batch, j));
            exp_sum += probs.at(batch, j);
        }

        for (size_t j = 0; j < probs.shape[1]; j++) {
            probs.at(batch, j) /= exp_sum;
        }
    }
}

// probs - batch_size x fan_out
// preds - batch_size x 1
void predictions(const Matrix& probs, Matrix& preds) {
    for (size_t batch = 0; batch < probs.shape[0]; batch++) {
        size_t highest_j = 0;
        for (size_t j = 0; j < probs.shape[1]; j++) {
            if (probs.at(batch, j) > probs.at(batch, highest_j)) {
                highest_j = j;
            }
        }
        preds.at(batch, 0) = highest_j;
    }
}

DT cross_entropy_from_probs(const Matrix& probs, const Matrix& targets) {
    DT loss = 0;
    for (size_t batch = 0; batch < targets.shape[0]; batch++) {
        size_t correct_i = targets.at(batch, 0);  // Implicit conversion
        loss += -logf(probs.at(batch, correct_i));
    }
    loss /= targets.shape[0];  // We want mean NLL

    return loss;
}
