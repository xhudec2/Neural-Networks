#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "../matrix/matrix.hpp"

void softmax(const Matrix& logits, Matrix& probs);
void predictions(const Matrix& probs, Matrix& preds);
DT cross_entropy_from_probs(const Matrix& probs, const Matrix& targets);

#endif