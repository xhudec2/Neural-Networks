#include "csv_reader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

void CSV::load(std::string filename) {
    std::fstream file(filename);
    std::string line;
    size_t index = 0;
    while (std::getline(file, line)) {
        std::string num;
        std::istringstream input(line);
        while(std::getline(input, num, ',')) {
            data.data[index] = std::stoi(num);
            ++index;
        }
    }
}
