#ifndef TESTS_HPP
#define TESTS_HPP

#include <vector>
#include "matrix.hpp"

Matrix full(const std::vector<size_t>& shape, DT val);
Matrix zeros(const std::vector<size_t>& shape);
Matrix identity(const std::vector<size_t>& shape);
Matrix iota(const std::vector<size_t>& shape);
Matrix random_normal(const std::vector<size_t>& shape, DT mean, DT std);
void tests();

#endif
