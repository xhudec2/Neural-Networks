#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include <sstream>
#include "printer.hpp"

struct BadShapeException : std::exception {
   private:
    std::string message;

   public:
    BadShapeException(std::string_view op, const shape_t& shape1,
                      const shape_t& shape2) {
        std::ostringstream oss;
        oss << "Invalid shapes for operation: " << op << ", ";
        oss << "Shape 1: " << shape1 << ", Shape 2: " << shape2;
        message = oss.str();
    }

    BadShapeException(std::string_view op, const shape_t& shape1,
                      const shape_t& shape2, const shape_t& shape3) {
        std::ostringstream oss;
        oss << "Invalid shapes for operation: " << op << ", ";
        oss << "Shape 1: " << shape1 << ", Shape 2: " << shape2
            << ", Shape 3: " << shape3;
        message = oss.str();
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#endif
