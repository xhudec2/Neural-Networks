#include "data/dataset.hpp"
#include "matrix/printer.hpp"
#include "tests/tests.hpp"
#include "parser/csv.hpp"
#include "constants.hpp"
#include "network/network.hpp"
#include "network/optimizer.hpp"
#include "hparams.hpp"


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

    Hparams hparams = {
        .shape = {2, 16, 16, 2},
        .learning_rate = 0.1,
        .num_epochs = 20,
        .batch_size = 4,
    };
    
    SGD optimizer(hparams.learning_rate);

    Network net(hparams.shape, optimizer);
    net.train(hparams);

    return 0;
}
