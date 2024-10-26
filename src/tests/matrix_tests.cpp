#include <algorithm>
#include <cassert>

#include "tests.hpp"
#include "../matrix/matrix.hpp"
#include "../matrix/printer.hpp"

void add_sub_test() {
    Matrix A = full({4, 3}, 2);
    Matrix B = full({4, 3}, 5);
    Matrix R = Matrix(A.shape);

    mat_add_const(A, 1, R);
    mat_sub_const(R, 1, R);
    assert(A.data == R.data);
}

void mul_div_test() {
    Matrix A = full({4, 3}, 2);
    Matrix B = full({4, 3}, 5);
    Matrix R = Matrix(A.shape);

    mat_mul_const(A, 4, R);
    mat_div_const(R, 4, R);
    assert(A.data == R.data);
}

void add_matrices_test() {
    Matrix A = full({4, 3}, 2);
    Matrix B = full({4, 3}, 5);
    Matrix R = Matrix(A.shape);

    mat_add_mat(A, B, R);
    assert(std::all_of(R.data.begin(), R.data.end(), [](DT a) { return a == 7; }));
}

void add_row_vec_test() {
    Matrix A = full({4, 3}, 2);
    Matrix R = Matrix(A.shape);
    std::vector<DT> vec{1, 2, 3};

    mat_add_row_vec(A, vec, R);

    for (size_t row = 0; row < R.shape[0]; row++) {
        for (size_t col = 0; col < R.shape[1]; col++) {
            assert(R.data[row * R.shape[1] + col] == 3 + col);
        }
    }
}

void add_col_vec_test() {
    Matrix A = full({4, 3}, 2);
    Matrix R = Matrix(A.shape);
    std::vector<DT> vec{1, 2, 3, 4};

    mat_add_col_vec(A, vec, R);
    for (size_t row = 0; row < R.shape[0]; row++) {
        for (size_t col = 0; col < R.shape[1]; col++) {
            assert(R.data[row * R.shape[1] + col] == 3 + row);
        }
    }
}

void mat_mul_vec_test() {
    Matrix A = full({4, 3}, 2);
    std::vector<DT> vec{1, 2, 3};
    std::vector<DT> r(4);

    mat_mul_vec(A, vec, r);
    assert(r[0] == 12 && r[1] == 12 && r[2] == 12 && r[3] == 12);
}

void mat_mul_mat_test() {
    Matrix C = iota({3, 3});
    Matrix Id = identity({3, 3});
    Matrix R = Matrix({3, 3});

    mat_mul_mat(C, Id, R);
    assert(C.data == R.data);
}

void mat_mul_rand_mat_test() {
    Matrix C = random_normal({5, 5}, 0, 1);
    Matrix Id = identity({5, 5});
    Matrix R = Matrix({5, 5});

    mat_mul_mat(C, Id, R);
    assert(C.data == R.data);
}

void mat_row_vec_multiply_operator() {
    shape_t shape = {5, 3};
    Matrix A = iota(shape);
    Matrix v = iota({1, 3});
    for (auto &val : v.data) {
        val += 1;
    }
    for (auto &val : A.data) {
        val += 1;
    }
    A *= v;
    assert((A.data == std::vector<DT>{{1, 2*2, 3*3, 4, 5*2, 6*3, 7, 8*2, 9*3, 10, 11*2, 12*3, 13, 14*2, 15*3}}));
}

void mat_col_vec_multiply_operator() {
    shape_t shape = {5, 3};
    Matrix A = iota(shape);
    Matrix v = iota({5, 1});
    for (auto &val : v.data) {
        val += 1;
    }
    for (auto &val : A.data) {
        val += 1;
    }
    A *= v;
    assert((A.data == std::vector<DT>{{1, 2, 3, 4*2, 5*2, 6*2, 7*3, 8*3, 9*3, 10*4, 11*4, 12*4, 13*5, 14*5, 15*5}}));
}



void matrix_tests() {
    add_sub_test();
    std::cout << "add_sub_test passed\n";
    mul_div_test();
    std::cout << "mul_div_test passed\n";
    add_matrices_test();
    std::cout << "add_matrices_test passed\n";
    add_row_vec_test();
    std::cout << "add_row_vec_test passed\n";
    add_col_vec_test();
    std::cout << "add_col_vec_test passed\n";
    mat_mul_vec_test();
    std::cout << "mat_mul_vec_test passed\n";
    mat_mul_mat_test();
    std::cout << "mat_mul_mat_test passed\n";
    mat_mul_rand_mat_test();
    std::cout << "mat_mul_rand_mat_test passed\n";
    mat_row_vec_multiply_operator();
    std::cout << "mat_row_vec_multiply_operator passed\n";
    mat_col_vec_multiply_operator();
    std::cout << "mat_col_vec_multiply_operator passed\n";
}

