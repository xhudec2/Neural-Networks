#include <sstream>
#include "matrix.hpp"
#include "exceptions.hpp"
#include "printer.hpp"

size_t product(const std::vector<size_t> &shape) {
    size_t result = 1;
    for (size_t v : shape) {
        result *= v;
    }
    return result;
}

void check_shape_eq(std::string str, const Matrix& A, const Matrix& R)
{
    if (A.shape != R.shape) {
        throw BadShapeException(str, A.shape, R.shape);
    }
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
