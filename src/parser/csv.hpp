#ifndef CSV_READER_HPP
#define CSV_READER_HPP

#include <string>
#include "../matrix/matrix.hpp"

struct CSV {
    Matrix data;

    CSV(const std::vector<size_t> &shape, std::string filename) : data{shape} {
        load(filename);
    }

    void load(std::string filename);
    void save(std::string filename) const;
};

#endif
