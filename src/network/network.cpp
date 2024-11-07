#include "network.hpp"

#include "../matrix/printer.hpp"

auto XOR_dataset() {
    Matrix inputs({4, 2}, {0, 0,
                           0, 1,
                           1, 0,
                           1, 1});
    Matrix targets({4, 1}, {0, 1, 1, 0});  // Each value is the index of the correct class

    return std::pair{inputs, targets};
}

void Network::train(Hparams hparams) {
    for (auto &layer : layers) {
        layer.prepare_layer(hparams.batch_size);
    }
    auto [inputs, targets] = XOR_dataset();
    Matrix outputs{{hparams.batch_size, layers.back().shape[1]}};
    for (size_t epoch = 0; epoch < hparams.num_epochs; ++epoch) {
        // print("--------------------");
        forward(inputs, outputs);
        backward(outputs, targets);
        update();
    }
}

void Network::forward(const Matrix &input, Matrix &outputs) {
    for (size_t i = 0; i < layers.size(); ++i) {
        bool last = i == layers.size() - 1;
        const Matrix &in = (i == 0) ? input : layers[i].inputs;
        Matrix &out = last ? outputs : layers[i + 1].inputs;

        layers[i].forward(in, out);
    }
}

Matrix cross_entropy(const Matrix &outputs, const Matrix &targets) {
    Matrix probs(outputs.shape);

    // Softmax for each batch
    for (size_t batch = 0; batch < outputs.shape[0]; batch++) {
        DT exp_sum = 0;
        for (size_t j = 0; j < outputs.shape[1]; j++) {
            probs[{batch, j}] = expf(outputs[{batch, j}]);
            exp_sum += probs[{batch, j}];
        }

        for (size_t j = 0; j < outputs.shape[1]; j++) {
            probs[{batch, j}] /= exp_sum;
        }
    }

    DT loss = 0;
    for (size_t batch = 0; batch < targets.shape[0]; batch++) {
        size_t correct_i = targets[{batch, 0}];  // Implicit conversion
        loss += -logf(probs[{batch, correct_i}]);
    }
    loss /= targets.shape[0];  // We want mean NLL
    print("loss:", " ");
    print(loss);

    Matrix dE_dy(probs.shape);

    // dE_dy = (predicted probs) - (one hot encoded targets)
    // (copy probs and subtract 1 from the index of the correct class in each
    // batch)
    dE_dy.data = probs.data;
    for (size_t batch = 0; batch < probs.shape[0]; batch++) {
        size_t correct_i = targets[{batch, 0}];
        dE_dy[{batch, correct_i}] = probs[{batch, correct_i}] - 1;
    }

    dE_dy /= static_cast<DT>(probs.shape[0]);  // Averaging over batch size

    return dE_dy;
}

void Network::backward(const Matrix &outputs, const Matrix &targets) {
    Matrix dE_dy = cross_entropy(outputs, targets);
    for (int i = layers.size() - 1; i >= 0; --i) {
        bool last = i == 0;
        layers[i].backward(dE_dy, last);
    }
}

void Network::update() {
    for (auto &layer : layers) {
        optimizer.step(layer.weights, layer.grad);
        optimizer.step(layer.bias, layer.bias_grad);
    }
}