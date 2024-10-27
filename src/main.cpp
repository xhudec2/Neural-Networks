#include "data/dataset.hpp"
#include "matrix/printer.hpp"
#include "tests/tests.hpp"
#include "parser/csv.hpp"
#include "constants.hpp"
#include "network/network.hpp"

int main() {
    // matrix_tests();
    // csv_tests();

    // Dataset ds(TRAIN_VEC_PATH, 512, 53760);
    // size_t batch_size = 1024;
    // Matrix batch({batch_size, 28 * 28});

    // for (size_t epoch = 0; epoch < 100; ++epoch) {
    //     std::cout << "epoch: " << epoch << '\n';
    //     for (size_t x = 0; x < 53760 / batch_size; ++x) {
    //         ds.get_next_batch(x * batch_size, false, batch);
    //     }
    // }
    Network net({2, 3, 5, 3, 1});
    // for (auto& layer : net.layers) {
    //     for (auto &weight : layer.weights.data) {
    //         weight = std::round(weight * 1000) / 1000;
    //     }
    //     for (auto &weight : layer.bias.data) {
    //         weight = std::round(weight * 1000) / 1000;
    //     }
    // }
    net.train();

    return 0;
}
