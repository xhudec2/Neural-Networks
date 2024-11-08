#include "network.hpp"

#include "../matrix/printer.hpp"
#include "helpers.hpp"

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

DT cross_entropy(const Matrix &outputs, const Matrix &targets, Matrix& dE_dy) {
    std::cout << "probs alloc\n";
    Matrix probs(outputs.shape);

    softmax(outputs, probs);
    DT loss = cross_entropy_from_probs(probs, targets);

    std::cout << "dE_dy alloc\n";
    dE_dy = Matrix(probs.shape);

    // dE_dy = (predicted probs) - (one hot encoded targets)
    // (copy probs and subtract 1 from the index of the correct class in each batch)
    dE_dy.data = probs.data;
    for (size_t batch = 0; batch < probs.shape[0]; batch++) {
        size_t correct_i = targets.at(batch, 0);
        dE_dy.at(batch, correct_i) = probs.at(batch, correct_i) - 1;
    }

    dE_dy /= static_cast<DT>(probs.shape[0]);  // Averaging over batch size

    return loss;
}

void Network::backward(const Matrix &outputs, const Matrix &targets) {
    Matrix dE_dy;
    loss = cross_entropy(outputs, targets, dE_dy);
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