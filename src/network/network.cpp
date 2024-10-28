#include "network.hpp"
#include "../matrix/printer.hpp"

auto XOR_dataset() {
    shape_t shape = {1, 2};
    Matrix a(shape); a.data = {1, 1};
    Matrix b(shape); b.data = {1, 0};
    Matrix c(shape); c.data = {0, 1};
    Matrix d(shape); d.data = {0, 0};
    std::vector<Matrix> inputs = {a, b, c, d};

    shape_t target_shape = {1, 1};
    Matrix target_a(target_shape); target_a.data = {1};
    Matrix target_b(target_shape); target_b.data = {0};
    Matrix target_c(target_shape); target_c.data = {0};
    Matrix target_d(target_shape); target_d.data = {1};
    std::vector<Matrix> targets = {target_a, target_b, target_c, target_d};

    return std::pair{inputs, targets};
}

void Network::train(Hparams hparams) {
    for (auto &layer : layers) {
        layer.prepare_layer(hparams.batch_size);
    }
    auto [inputs, targets] = XOR_dataset();
    Matrix outputs{targets[0].shape};
    for (size_t epoch = 0; epoch < hparams.num_epochs; ++epoch) {
        print("--------------------");
        for (size_t i = 0; i < targets.size(); ++i) {
            forward(inputs[i], outputs);
            backward(outputs, targets[i]);
            update(hparams.learning_rate);
        }
    }
}

void Network::forward(const Matrix &input, Matrix &outputs) {
    for (size_t i = 0; i < layers.size(); ++i) {
        const Matrix &in = (i == 0) ? input : layers[i].inputs;
        Matrix &out = (i == layers.size() - 1) ? outputs : layers[i + 1].inputs;
        layers[i].forward(in, out);
    }
}

Matrix get_loss(const Matrix &output, const Matrix &target) {
    Matrix loss(target.shape);

    std::cout << "target: " << target.data[0] << "; ";
    std::cout << "output: " << output.data[0] << "; ";
    std::cout << "loss: ";
    DT denom = target.data[0] == 0 ? (output.data[0] - 1) : (output.data[0]);
    loss.data[0] = -1.0 / (denom + 0.01);
    if (target.data[0] == 1) {
        std::cout << -log(static_cast<double>(output.data[0] + 0.00001));
    } else {
        std::cout << -log(1.0 - static_cast<double>(output.data[0] + 0.00001));
    }
    std::cout << "\n";

    return loss;
}
    
void Network::backward(const Matrix &output, const Matrix &target) {
    Matrix dE_dy = get_loss(output, target);
    for (int i = layers.size() - 1; i >= 0; --i) {
        bool last = i == 0;
        layers[i].backward(dE_dy, last);
    }
}

void Network::update(float lr) {
    for (auto &layer : layers) {
        layer.gradient_accum *= -lr;
        layer.weights += layer.gradient_accum;
        layer.gradient_accum *= 0;

        layer.bias_gradient_accum *= -lr;
        layer.bias += layer.bias_gradient_accum;
        layer.bias_gradient_accum *= 0;
    }
}