#include "constants.hpp"
#include "data/dataset.hpp"
#include "hparams.hpp"
#include "matrix/printer.hpp"
#include "network/network.hpp"
#include "network/optimizer.hpp"
#include "network/helpers.hpp"
#include "parser/csv.hpp"
#include "tests/tests.hpp"

int main() {
    // matrix_tests();
    // csv_tests();

    Hparams hparams = {
        .shape = {IMG_SIZE, 32, 32, 10},
        .learning_rate = 0.1,
        .num_epochs = 1,
        .batch_size = 128,
    };

    Dataset ds(TRAIN_VEC_PATH, TRAIN_LABEL_PATH, hparams.batch_size, TRAIN_SIZE);

    // TODO: Replace with actual normalization
    ds.Xdata /= 255.;

    SGD optimizer(hparams.learning_rate);
    Network net(hparams.shape, optimizer);

    Matrix Xbatch({hparams.batch_size, IMG_SIZE});
    Matrix ybatch({hparams.batch_size, 1});

    for (auto &layer : net.layers) {
        layer.prepare_layer(hparams.batch_size);
    }
    // auto [inputs, targets] = XOR_dataset();
    Matrix outputs{{hparams.batch_size, net.layers.back().shape[1]}};
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
        
        // TODO: VAL_SIZE size not divisible by batch_size
        for (size_t batch = 0; batch < VAL_SIZE / hparams.batch_size; ++batch) {
            ds.get_next_batch(batch * hparams.batch_size, true, Xbatch, ybatch);
            net.forward(Xbatch, outputs);
            Matrix probs(outputs.shape);
            softmax(outputs, probs);
            Matrix preds(ybatch.shape);
            predictions(probs, preds);
            loss += cross_entropy_from_probs(probs, ybatch);

            total_preds += hparams.batch_size;
            for (size_t b = 0; b < hparams.batch_size; b++) {
                if (preds[{b, 0}] == ybatch[{b, 0}]) {
                    correct_preds++;
                }
            }
        }
        
        loss /= ((double)VAL_SIZE / hparams.batch_size);
        
        print("Val loss: ", "");
        print(loss);
        print("Val acc.: ", "");
        print((double) correct_preds / total_preds);
    }

    return 0;
}
