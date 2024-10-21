#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include "printer.hpp"
#include <exception>
#include <sstream>
#include <vector>
#include <string_view>

struct BadShapeException : std::exception {
   private:
    std::string message;

   public:
    BadShapeException(std::string_view op, const std::vector<size_t>& shape1,
                      const std::vector<size_t>& shape2) {
        std::ostringstream oss;
        oss << "Invalid shapes for operation: " << op << ", ";
        oss << "Shape 1: " << shape1 << ", Shape 2: " << shape2;
        message = oss.str();
    }

    BadShapeException(std::string_view op, const std::vector<size_t>& shape1,
                      const std::vector<size_t>& shape2,
                      const std::vector<size_t>& shape3) {
        std::ostringstream oss;
        oss << "Invalid shapes for operation: " << op << ", ";
        oss << "Shape 1: " << shape1 << ", Shape 2: " << shape2
            << ", Shape 3: " << shape3;
        message = oss.str();
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#endif
