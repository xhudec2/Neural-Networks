#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>

using DT = float;

size_t product(const std::vector<size_t> &shape);

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
    // friend Matrix operator+(Matrix A, Matrix B);
    // friend Matrix operator+(Matrix A, DT c);
    // friend Matrix operator*(Matrix A, DT c);
    // friend Matrix operator*(Matrix A, Matrix B);
    // friend Matrix operator-(Matrix A, DT c);
    // friend Matrix operator-(Matrix A, Matrix B);
    // friend Matrix operator/(Matrix A, DT c);
};

void mat_add_const(const Matrix& A, DT c, Matrix& R);
void mat_sub_const(const Matrix& A, DT c, Matrix& R);
void mat_mul_const(const Matrix& A, DT c, Matrix& R);
void mat_div_const(const Matrix& A, DT c, Matrix& R);
void mat_add_mat(const Matrix& A, const Matrix& B, Matrix& R);
void mat_add_row_vec(const Matrix& A, const std::vector<DT>& b, Matrix& R);
void mat_add_col_vec(const Matrix& A, const std::vector<DT>& b, Matrix& R);
void mat_mul_vec(const Matrix& A, const std::vector<DT>& b, std::vector<DT>& r);
void mat_mul_mat(const Matrix& A, const Matrix& B, Matrix& R);

Matrix full(const std::vector<size_t>& shape, DT val);
Matrix zeros(const std::vector<size_t>& shape);
Matrix identity(const std::vector<size_t>& shape);
Matrix iota(const std::vector<size_t>& shape);
Matrix random_normal(const std::vector<size_t>& shape, DT mean, DT std);

#endif
