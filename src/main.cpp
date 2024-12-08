#include "data/dataset.hpp"
#include "matrix/printer.hpp"
#include "network/helpers.hpp"
#include "network/network.hpp"
#include "tests/tests.hpp"


struct Hparams {
    shape_t shape;
    float learning_rate;
    size_t num_epochs;
    size_t batch_size;
    bool shuffle;
};

Hparams hparams = {
    .shape = {IMG_SIZE, 256, 128, 10},
    .learning_rate = 0.0011,
    .num_epochs = 15,
    .batch_size = 100,
    .shuffle = true,
};

int main() {
    // Try out these tests :)
    // matrix_tests();
    // csv_tests();

    std::cout << "seed: " << RAND_SEED << "\n";
    Dataset ds(TRAIN_VEC_PATH, TRAIN_LABEL_PATH, hparams.batch_size,
               TRAIN_SIZE);

    // shuffles the entire dataset, the train and the val part together
    if (hparams.shuffle) {
        ds.shuffle(true);
    }

    // mean and std calculated only on train portion of train dataset, val is left out
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
        if (hparams.shuffle) {
            // shuffles only train part of the dataset
            ds.shuffle(false);
        }
        print("--------------------");
        print("Epoch ", "");
        print(epoch);

        for (size_t batch = 0; batch < TRAIN_SIZE / hparams.batch_size;
             ++batch) {
            ds.get_next_batch(batch * hparams.batch_size, false, Xbatch,
                              ybatch);
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
            argmax(probs, preds);
            loss += cross_entropy_from_probs(probs, ybatch);

            total_preds += hparams.batch_size;
            for (size_t b = 0; b < hparams.batch_size; b++) {
                if (preds.at(b, 0) == ybatch.at(b, 0)) {
                    correct_preds++;
                }
            }
        }

        loss /= (static_cast<double>(VAL_SIZE) / hparams.batch_size);

        print("Val loss: ", "");
        print(loss);
        print("Val acc.: ", "");
        print(static_cast<double>(correct_preds) / total_preds);
        if (epoch > 1 && prev_loss < loss) {
            optimizer.learning_rate *= 0.5;
            print();
            print("LR decay: ", "");
            print(optimizer.learning_rate);
        }
        prev_loss = loss;
    }
    
    {
        net.prepare(DATASET_SIZE);

        Matrix Xdata({DATASET_SIZE, IMG_SIZE});
        Matrix ydata({DATASET_SIZE, 1});
        CSV::load(Xdata, TRAIN_VEC_PATH);
        CSV::load(ydata, TRAIN_LABEL_PATH);
        Xdata /= 255.;
        for (auto &pixel : Xdata.data) {
            pixel -= mean;
            pixel /= std;
        }
        Matrix out({DATASET_SIZE, 10});
        Matrix probs({DATASET_SIZE, 10});
        Matrix preds({DATASET_SIZE, 1});
        net.forward(Xdata, out, true);
        softmax(out, probs);
        argmax(probs, preds);
        DT loss = cross_entropy_from_probs(probs, ydata);

        print("Train loss: ", "");
        print(loss);

        CSV::save(preds, "train_predictions.csv");
    }

    {
        net.prepare(TEST_SIZE);

        Matrix Xtest({TEST_SIZE, IMG_SIZE});
        Matrix ytest({TEST_SIZE, 1});
        CSV::load(Xtest, TEST_VEC_PATH);
        CSV::load(ytest, TEST_LABEL_PATH);
        Xtest /= 255.;
        for (auto &pixel : Xtest.data) {
            pixel -= mean;
            pixel /= std;
        }
        Matrix test_out({TEST_SIZE, 10});
        Matrix probs({TEST_SIZE, 10});
        Matrix preds({TEST_SIZE, 1});
        net.forward(Xtest, test_out, true);
        softmax(test_out, probs);
        argmax(probs, preds);
        DT loss = cross_entropy_from_probs(probs, ytest);

        print("Test loss: ", "");
        print(loss);

        CSV::save(preds, "test_predictions.csv");
    }

    return 0;
}
