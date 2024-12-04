#include <cassert>

#include "../constants.hpp"
#include "../parser/csv.hpp"

void test_read_write_csv(const std::vector<size_t>& shape,
                         const std::string& file_path) {
    const std::string CSV_TEST_PATH = "./csv_test_file.csv";

    Matrix m(shape);
    CSV::load(m, file_path);
    CSV::save(m, CSV_TEST_PATH);

    std::string diff_command = "diff " + file_path + ' ' + CSV_TEST_PATH;
    assert(system(diff_command.c_str()) == 0);

    std::string rm_command = "rm " + CSV_TEST_PATH;
    assert(system(rm_command.c_str()) == 0);  // Just to stop a warning
}

void csv_tests() {
    test_read_write_csv({DATASET_SIZE, IMG_SIZE}, TRAIN_VEC_PATH);
    std::cout << "Reading and writing train vectors passed\n";
    test_read_write_csv({DATASET_SIZE, 1}, TRAIN_LABEL_PATH);
    std::cout << "Reading and writing train labels passed\n";
    test_read_write_csv({TEST_SIZE, IMG_SIZE}, TEST_VEC_PATH);
    std::cout << "Reading and writing test vectors passed\n";
    test_read_write_csv({TEST_SIZE, 1}, TEST_LABEL_PATH);
    std::cout << "Reading and writing test labels passed\n";
}