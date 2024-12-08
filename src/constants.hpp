#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <string>
#include <ctime>
#include <random>

// const unsigned int RAND_SEED = static_cast<unsigned int>(std::time(nullptr));
const unsigned int RAND_SEED = 42;
extern std::random_device rd;
extern std::mt19937 gen;

const std::string DATA_PATH = "./data/";
const std::string TRAIN_VEC_PATH = DATA_PATH + "fashion_mnist_train_vectors.csv";
const std::string TRAIN_LABEL_PATH = DATA_PATH + "fashion_mnist_train_labels.csv";
const std::string TEST_VEC_PATH = DATA_PATH + "fashion_mnist_test_vectors.csv";
const std::string TEST_LABEL_PATH = DATA_PATH + "fashion_mnist_test_labels.csv";
const size_t IMG_SIZE = 28 * 28;
const size_t DATASET_SIZE = 60000;
const size_t TRAIN_SIZE = 50000;
const size_t VAL_SIZE = DATASET_SIZE - TRAIN_SIZE;
const size_t TEST_SIZE = 10000;

#endif