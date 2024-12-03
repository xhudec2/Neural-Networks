#include "csv.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

void CSV::load(Matrix& M, std::string filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    std::string line;
    size_t i = 0;
    while (std::getline(file, line)) {
        const char* ptr = line.c_str();
        int value = 0;

        while (*ptr) {
            if (*ptr == ',') {
                M.data[i++] = value;
                value = 0;
            } else {
                value = value * 10 + (*ptr - '0');
            }
            ptr++;
        }

        M.data[i++] = value;
    }
}

void CSV::save(const Matrix& M, std::string filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    for (size_t row = 0; row < M.shape[0]; row++) {
        for (size_t col = 0; col < M.shape[1]; col++) {
            file << static_cast<int>(M.data[row * M.shape[1] + col]);
            if (col != M.shape[1] - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}