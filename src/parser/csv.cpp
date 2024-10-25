#include "csv.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

void CSV::load(std::string filename) {
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
                data.data[i++] = value;
                value = 0;
            } else {
                value = value * 10 + (*ptr - '0');
            }
            ptr++;
        }

        data.data[i++] = value;
    }
}

void CSV::save(std::string filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    for (size_t row = 0; row < data.shape[0]; row++) {
        for (size_t col = 0; col < data.shape[1]; col++) {
            file << static_cast<int>(data.data[row * data.shape[1] + col]);
            if (col != data.shape[1] - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}