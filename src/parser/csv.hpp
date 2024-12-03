#ifndef CSV_READER_HPP
#define CSV_READER_HPP

#include <string>

#include "../matrix/matrix.hpp"

struct CSV {
    Matrix data;

    static void load(Matrix& m, std::string filename);
    static void save(const Matrix& m, std::string filename);
};

#endif
