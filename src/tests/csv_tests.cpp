#include <cassert>

#include "../constants.hpp"
#include "../parser/csv.hpp"

void test_read_write_csv(const std::vector<size_t>& shape,
                         const std::string& file_path) {
    const std::string CSV_TEST_PATH = "./csv_test_file.csv";

    CSV csv(shape, file_path);
    csv.save(CSV_TEST_PATH);

    std::string diff_command = "diff " + file_path + ' ' + CSV_TEST_PATH;
    assert(system(diff_command.c_str()) == 0);

    std::string rm_command = "rm " + CSV_TEST_PATH;
    assert(system(rm_command.c_str()) == 0); // Assert here just to stop a warning
}

void csv_tests() {
    test_read_write_csv({60000, 784}, TRAIN_VEC_PATH);
    std::cout << "Reading and writing train vectors passed\n";
    test_read_write_csv({60000, 1}, TRAIN_LABEL_PATH);
    std::cout << "Reading and writing train labels passed\n";
    test_read_write_csv({10000, 784}, TEST_VEC_PATH);
    std::cout << "Reading and writing test vectors passed\n";
    test_read_write_csv({10000, 1}, TEST_LABEL_PATH);
    std::cout << "Reading and writing test labels passed\n";
}