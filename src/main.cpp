#include <cstddef>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << '{';
    for (size_t i = 0; i < v.size(); i++) {
        os << v[i];
        if (i != v.size() - 1) {
            os << ", ";
        }
    }
    os << '}';

    return os;
}

class BadShapeException : public std::exception {
   private:
    std::string message;

   public:
    BadShapeException(std::string op, std::vector<size_t> shape1,
                      std::vector<size_t> shape2) {
        std::ostringstream oss;
        oss << "Invalid shapes for operation: " << op << ", ";
        oss << "Shape 1: " << shape1 << ", Shape 2: " << shape2;
        message = oss.str();
    }

    BadShapeException(std::string op, std::vector<size_t> shape1,
                      std::vector<size_t> shape2, std::vector<size_t> shape3) {
        std::ostringstream oss;
        oss << "Invalid shapes for operation: " << op << ", ";
        oss << "Shape 1: " << shape1 << ", Shape 2: " << shape2
            << ", Shape 3: " << shape3;
        message = oss.str();
    }

    const char* what() const noexcept override { return message.c_str(); }
};

template <typename T>
T product(const std::vector<T>& v) {
    T result = 1;
    for (T item : v) {
        result *= item;
    }

    return result;
}

using DT = float;

struct Matrix {
    std::vector<DT> data;
    std::vector<size_t> shape;
    size_t size;

    Matrix(std::vector<size_t> shape) : shape{shape} {
        size = product(shape);
        data = std::vector<DT>(size);
    }
    Matrix(std::vector<size_t> shape, DT val) : shape{shape} {
        size = product(shape);
        data = std::vector<DT>(size, val);
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& M) {
        os << '[';
        for (size_t row = 0; row < M.shape[0]; row++) {
            if (row > 0) {
                os << ' ';
            }
            os << '[';
            for (size_t col = 0; col < M.shape[1]; col++) {
                os << M.data[row * M.shape[1] + col];
                if (col != M.shape[1] - 1) {
                    os << ", ";
                }
            }
            os << ']';
            if (row != M.shape[0] - 1) {
                os << '\n';
            }
        }

        os << "]\n";

        return os;
    }
};

#define check_shape_eq(str, A, R)                       \
    if (A.shape != R.shape) {                           \
        throw BadShapeException(str, A.shape, R.shape); \
    }

void mat_add_const(const Matrix& A, DT c, Matrix& R) {
    check_shape_eq("mat_add_const (result)", A, R);

    for (size_t i = 0; i < A.size; i++) {
        R.data[i] = A.data[i] + c;
    }
}

void mat_sub_const(const Matrix& A, DT c, Matrix& R) {
    check_shape_eq("mat_sub_const (result)", A, R);

    for (size_t i = 0; i < A.size; i++) {
        R.data[i] = A.data[i] - c;
    }
}

void mat_mul_const(const Matrix& A, DT c, Matrix& R) {
    check_shape_eq("mat_mul_const (result)", A, R);

    for (size_t i = 0; i < A.size; i++) {
        R.data[i] = A.data[i] * c;
    }
}

void mat_div_const(const Matrix& A, DT c, Matrix& R) {
    if (c == 0) {
        throw std::runtime_error("mat_div_const: Division by zero");
    }
    check_shape_eq("mat_div_const (result)", A, R);

    for (size_t i = 0; i < A.size; i++) {
        R.data[i] = A.data[i] / c;
    }
}

void mat_add_mat(const Matrix& A, const Matrix& B, Matrix& R) {
    if (A.shape != B.shape) {
        throw BadShapeException("mat_add_mat", A.shape, B.shape);
    }
    check_shape_eq("mat_add_mat (result)", A, R);

    for (size_t i = 0; i < A.size; i++) {
        R.data[i] = A.data[i] + B.data[i];
    }
}

void mat_add_row_vec(const Matrix& A, const std::vector<DT>& b, Matrix& R) {
    if (A.shape[1] != b.size()) {
        throw BadShapeException("mat_add_row_vec", A.shape, {b.size()});
    }
    check_shape_eq("mat_add_row_vec (result)", A, R);

    for (size_t row = 0; row < A.shape[0]; row++) {
        for (size_t col = 0; col < A.shape[1]; col++) {
            R.data[row * A.shape[1] + col] =
                A.data[row * A.shape[1] + col] + b[col];
        }
    }
}

void mat_add_col_vec(const Matrix& A, const std::vector<DT>& b, Matrix& R) {
    if (A.shape[0] != b.size()) {
        throw BadShapeException("mat_add_col_vec", A.shape, {b.size()});
    }
    check_shape_eq("mat_add_col_vec (result)", A, R);

    for (size_t row = 0; row < A.shape[0]; row++) {
        for (size_t col = 0; col < A.shape[1]; col++) {
            R.data[row * A.shape[1] + col] =
                A.data[row * A.shape[1] + col] + b[row];
        }
    }
}

void mat_mul_vec(const Matrix& A, const std::vector<DT>& b,
                 std::vector<DT>& r) {
    if (A.shape[1] != b.size()) {
        throw BadShapeException("mat_mul_vec", A.shape, {b.size()});
    }
    if (A.shape[0] != r.size()) {
        throw BadShapeException("mat_mul_vec (result)", A.shape, {r.size()});
    }

    memset(r.data(), 0, r.size() * sizeof(DT));
    for (size_t row = 0; row < A.shape[0]; row++) {
        DT dot_product = 0;
        for (size_t col = 0; col < A.shape[1]; col++) {
            dot_product += A.data[row * A.shape[1] + col] * b[col];
        }
        r[row] = dot_product;
    }
}

void mat_mul_mat(const Matrix& A, const Matrix& B, Matrix& R) {
    if (A.shape[1] != B.shape[0]) {
        throw BadShapeException("mat_mul_mat", A.shape, B.shape);
    }
    if (A.shape[0] != R.shape[0] || B.shape[1] != R.shape[1]) {
        throw BadShapeException("mat_mul_mat (result)", A.shape, B.shape,
                                R.shape);
    }

    memset(R.data.data(), 0, R.data.size() * sizeof(DT));

    for (size_t k = 0; k < B.shape[0]; k++) {
        for (size_t i = 0; i < A.shape[0]; i++) {
            for (size_t j = 0; j < B.shape[1]; j++) {
                R.data[i * R.shape[1] + j] +=
                    A.data[i * A.shape[1] + k] * B.data[k * B.shape[1] + j];
            }
        }
    }
}

Matrix full(std::vector<size_t> shape, DT val) { return Matrix(shape, val); }
Matrix zeros(std::vector<size_t> shape) { return full(shape, 0); }
Matrix identity(std::vector<size_t> shape) {
    Matrix result = zeros(shape);
    for (size_t i = 0; i < shape[0]; i++) {
        result.data[i * shape[1] + i] = 1;
    }

    return result;
}

void tests() {
    Matrix A = full({4, 3}, 2);
    Matrix B = full({4, 3}, 5);
    Matrix R = Matrix(A.shape);

    mat_add_const(A, 1, R);
    std::cout << R << '\n';

    mat_add_mat(A, B, R);
    std::cout << R << '\n';

    std::vector<DT> v{1, 2, 3};

    mat_add_row_vec(A, v, R);
    std::cout << R << '\n';

    v = {1, 2, 3, 4};

    mat_add_col_vec(A, v, R);
    std::cout << R << '\n';

    v = {1, 2, 3};
    std::vector<DT> r(4);

    mat_mul_vec(A, v, r);
    std::cout << r << "\n\n";

    Matrix Id = identity({3, 3});
    R = Matrix({4, 3});

    mat_mul_mat(A, Id, R);
    std::cout << R << '\n';

    A = full({1000, 1000}, 2);
    B = full({1000, 1000}, 3);
    R = Matrix(A.shape);

    mat_mul_mat(A, B, R);
}

int main() {
    tests();
    return 0;
}