#include "constants.hpp"
#include "data/dataset.hpp"
#include "hparams.hpp"
#include "matrix/printer.hpp"
#include "network/network.hpp"
#include "network/optimizer.hpp"
#include "network/helpers.hpp"
#include "parser/csv.hpp"
#include "tests/tests.hpp"
#include <memory>

// void *operator new (size_t size) {
//     std::cout << "Allocating " << size << " bytes\n";

//     return malloc(size);
// }

// void operator delete (void *memory) {
//     std::cout << "freeing " << memory << "\n";

//     free(memory);
// }

int main() {
    // matrix_tests();
    // csv_tests();
    std::cout << "seed: " << RAND_SEED << "\n";
    bool test_run = true;
    Hparams hparams = {
        .shape = {IMG_SIZE, 256, 256, 10},
        .learning_rate = 0.001,
        .num_epochs = 5,
        .batch_size = 100,
    };

    Dataset ds(TRAIN_VEC_PATH, TRAIN_LABEL_PATH, hparams.batch_size, TRAIN_SIZE);

    ds.Xdata /= 255.;
    double mean = 0.0;
    for (size_t i = 0; i < TRAIN_SIZE * IMG_SIZE; ++i) {
        mean += ds.Xdata.data[i];
    }
    mean /= TRAIN_SIZE * IMG_SIZE;
    double std = 0.0;
    for (auto &pixel : ds.Xdata.data) {
        pixel -= mean;
    }
    for (size_t i = 0; i < TRAIN_SIZE * IMG_SIZE; ++i) {
        std += ds.Xdata.data[i] * ds.Xdata.data[i];
    }
    std /= (TRAIN_SIZE * IMG_SIZE - 1);
    std = sqrtf(std);
    std::cout << "mean: " << mean << '\n';
    std::cout << "sample std: " << std << '\n';
    for (auto &pixel : ds.Xdata.data) {
        pixel /= std;
    }

    Adam optimizer(hparams.learning_rate);
    Network net(hparams.shape, optimizer);
    net.prepare(hparams.batch_size);

    Matrix Xbatch({hparams.batch_size, IMG_SIZE});
    Matrix ybatch({hparams.batch_size, 1});
    Matrix outputs{{hparams.batch_size, net.layers.back().shape[1]}};
    
    DT prev_loss = 0;
    for (size_t epoch = 1; epoch <= hparams.num_epochs; ++epoch) {
        print("--------------------");
        print("Epoch ", "");
        print(epoch);

        for (size_t batch = 0; batch < TRAIN_SIZE / hparams.batch_size; ++batch) {
            ds.get_next_batch(batch * hparams.batch_size, false, Xbatch, ybatch);

            net.forward(Xbatch, outputs);
            net.backward(outputs, ybatch);
            net.update();
        }

        print("Train Loss: ", "");
        print(net.loss);

        size_t correct_preds = 0;
        size_t total_preds = 0; 
        DT loss = 0;
        
        Matrix probs(outputs.shape);
        Matrix preds(ybatch.shape);
        for (size_t batch = 0; batch < VAL_SIZE / hparams.batch_size; ++batch) {
            ds.get_next_batch(batch * hparams.batch_size, true, Xbatch, ybatch);
            net.forward(Xbatch, outputs, true);
            softmax(outputs, probs);
            predictions(probs, preds);
            loss += cross_entropy_from_probs(probs, ybatch);

            total_preds += hparams.batch_size;
            for (size_t b = 0; b < hparams.batch_size; b++) {
                if (preds.at(b, 0) == ybatch.at(b, 0)) {
                    correct_preds++;
                }
            }
        }
        
        loss /= ((double)VAL_SIZE / hparams.batch_size);
        
        print("Val loss: ", "");
        print(loss);
        print("Val acc.: ", "");
        print((double) correct_preds / total_preds);
        if (epoch > 1 && prev_loss < loss) {
            optimizer.learning_rate *= 0.5;
            print(optimizer.learning_rate);
        }
        prev_loss = loss;
    }

    if (test_run) {
        size_t correct_preds = 0;
        size_t total_preds = 0; 
        DT loss = 0;

        Dataset test_ds(TEST_VEC_PATH, TEST_LABEL_PATH, hparams.batch_size, TEST_SIZE);
        test_ds.Xdata /= 255.;
        for (auto &pixel : test_ds.Xdata.data) {
            pixel -= mean;
            pixel /= std;
        }
        Matrix probs(outputs.shape);
        Matrix preds(ybatch.shape);
        for (size_t batch = 0; batch < TEST_SIZE / hparams.batch_size; ++batch) {
            test_ds.get_next_batch(batch * hparams.batch_size, false, Xbatch, ybatch);
            net.forward(Xbatch, outputs, true);
            softmax(outputs, probs);
            predictions(probs, preds);
            loss += cross_entropy_from_probs(probs, ybatch);

            total_preds += hparams.batch_size;
            for (size_t b = 0; b < hparams.batch_size; b++) {
                if (preds.at(b, 0) == ybatch.at(b, 0)) {
                    correct_preds++;
                }
            }
        }
        
        loss /= ((double)TEST_SIZE / hparams.batch_size);
        
        print("Test loss: ", "");
        print(loss);
        print("Test acc.: ", "");
        print((double) correct_preds / total_preds);
    }
    return 0;
}
