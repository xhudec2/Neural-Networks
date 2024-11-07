#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <cassert>

using DT = float;
using shape_t = std::vector<size_t>;
size_t product(const shape_t& shape);

struct Matrix {
    shape_t shape;
    std::vector<DT> data;

    Matrix() {}

    Matrix(shape_t shape) : shape{shape} {
        data = std::vector<DT>(product(shape));
    }
    Matrix(shape_t shape, DT val) : shape{shape} {
        data = std::vector<DT>(product(shape), val);
    }
    Matrix(shape_t shape, std::initializer_list<DT> vals) : shape{shape}, 
                                                            data{vals} {}

    Matrix T() {
        assert(shape[0] == 1 or shape[1] == 1);
        Matrix mat = *this;
        std::swap(mat.shape[0], mat.shape[1]);
        return mat;
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

    Matrix &operator=(DT);
    Matrix &operator+=(const Matrix &);
    Matrix &operator/(DT);
    Matrix &operator/=(DT);
    Matrix &operator-(const Matrix &);
    Matrix &operator*=(const Matrix &);
    Matrix &operator*=(DT);
    size_t size() const { return data.size(); }
    
    const DT& operator[](shape_t indices) const;
    DT& operator[](shape_t i);
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

Matrix full(const shape_t& shape, DT val);
Matrix zeros(const shape_t& shape);
Matrix identity(const shape_t& shape);
Matrix iota(const shape_t& shape);
Matrix random_normal(const shape_t& shape, DT mean, DT std);

#endif
