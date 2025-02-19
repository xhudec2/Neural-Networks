#ifndef PRINTER_HPP
#define PRINTER_HPP

// Only for debugging

#include <iostream>
#include <vector>

inline std::ostream& operator<<(std::ostream& os, const shape_t& vec) {
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

template <typename T>
void print(T thing, std::string end = "\n") {
    std::cout << thing << end;
}
inline void print() { std::cout << '\n'; }

#endif
