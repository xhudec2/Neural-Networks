#include "data/dataset.hpp"
#include "matrix/printer.hpp"
#include "matrix/tests.hpp"
#include "parser/csv_reader.hpp"

int main() {
    tests();

    std::string path =
        "/Users/xhudec2/Documents/School/5. "
        "semester/PV021/Neural-Networks/data/fashion_mnist_train_vectors.csv";

    Dataset ds(path, 512, 53760);
    size_t batch_size = 1024;
    Matrix batch({batch_size, 28 * 28});

    for (size_t epoch = 0; epoch < 100; ++epoch) {
        std::cout << "epoch: " << epoch << '\n';
        for (size_t x = 0; x < 53760 / batch_size; ++x) {
            ds.get_next_batch(x * batch_size, false, batch);
        }
    }

    return 0;
}
