#include "network.hpp"

#include "../matrix/printer.hpp"
#include "helpers.hpp"

auto XOR_dataset() {
    Matrix inputs({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Matrix targets(
        {4, 1}, {0, 1, 1, 0});  // Each value is the index of the correct class

    return std::pair{inputs, targets};
}

void Network::train(Hparams hparams) {
    for (auto &layer : layers) {
        layer.prepare(hparams.batch_size);
    }
    auto [inputs, targets] = XOR_dataset();
    Matrix outputs{{hparams.batch_size, layers.back().shape[1]}};
    for (size_t epoch = 0; epoch < hparams.num_epochs; ++epoch) {
        forward(inputs, outputs);
        backward(outputs, targets);
        update();
    }
}

void Network::forward(const Matrix &input, Matrix &outputs, bool no_grad) {
    for (size_t i = 0; i < layers.size(); ++i) {
        bool last = i == layers.size() - 1;
        const Matrix &in = (i == 0) ? input : layers[i].inputs;
        Matrix &out = last ? outputs : layers[i + 1].inputs;

        layers[i].forward(in, out, no_grad);
    }
}

void Network::prepare(size_t batch_size) {
    probs = Matrix({batch_size, 10});
    dE_dy = Matrix({batch_size, 10});

    for (auto &layer : layers) {
        layer.prepare(batch_size);
    }
}

DT Network::cross_entropy(const Matrix &outputs, const Matrix &targets) {
    softmax(outputs, probs);
    DT loss = cross_entropy_from_probs(probs, targets);

    // dE_dy = (predicted probs) - (one hot encoded targets)
    // (copy probs and subtract 1 from the index of the correct class in each
    // batch)
    dE_dy.data = probs.data;
    for (size_t batch = 0; batch < probs.shape[0]; batch++) {
        size_t correct_i = targets.at(batch, 0);
        dE_dy.at(batch, correct_i) = probs.at(batch, correct_i) - 1;
    }

    dE_dy /= static_cast<DT>(probs.shape[0]);  // Averaging over batch size

    return loss;
}

void Network::backward(const Matrix &outputs, const Matrix &targets) {
    loss = cross_entropy(outputs, targets);
    layers[layers.size() - 1].backward(dE_dy, false);
    for (int i = layers.size() - 2; i >= 0; --i) {
        bool last = i == 0;
        layers[i].backward(layers[i + 1].dE_dOut, last);
    }
}

void Network::update() {
    for (auto &layer : layers) {
        optimizer.step(layer.weights, layer.grad, layer.momentum,
                       layer.rmsprop);
        optimizer.step(layer.bias, layer.bias_grad, layer.bias_momentum,
                       layer.bias_rmsprop);
    }
}