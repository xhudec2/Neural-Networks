#ifndef PRINTER_HPP
#define PRINTER_HPP

#include <iostream>
#include <vector>

template <typename T>
void print(T thing) {
    std::cout << thing << '\n';
}
inline void print() { std::cout << '\n'; }

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& vec) {
    os << '{';
    for (size_t i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << '}';
    return os;
}

#endif
