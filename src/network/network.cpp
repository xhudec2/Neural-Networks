#include "network.hpp"
#include "../matrix/printer.hpp"

auto XOR_dataset() {
    shape_t shape = {1, 2};
    Matrix a(shape); a.data = {1, 1};
    Matrix b(shape); b.data = {1, 0};
    Matrix c(shape); c.data = {0, 1};
    Matrix d(shape); d.data = {0, 0};
    std::vector<Matrix> inputs = {a, b, d, c};

    shape_t target_shape = {1, 1};
    Matrix target_a(target_shape); target_a.data = {0};
    Matrix target_b(target_shape); target_b.data = {1};
    Matrix target_c(target_shape); target_c.data = {1};
    Matrix target_d(target_shape); target_d.data = {0};
    std::vector<Matrix> targets = {target_a, target_b, target_d, target_c};

    return std::pair{inputs, targets};
}

void Network::train() {
    for (auto &layer : layers) {
        layer.prepare_layer(1);
    }
    auto [inputs, targets] = XOR_dataset();
    Matrix outputs{targets[0].shape};
    for (size_t epoch = 0; epoch < 1; ++epoch) {
        print("layers: ");
        for (auto &layer : layers) {
            print("weights: ");
            print(layer.weights);
            print("bias: ");
            print(layer.bias);
        }
        print("");
        for (size_t i = 2; i < targets.size(); ++i) {
            forward(inputs[i], outputs);
            backward(outputs, targets[i]);
            update();
            print("");
            break;
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

    std::cout << "target: " << target.data[0] << '\n';
    std::cout << "output: " << output.data[0] << '\n';
    std::cout << "loss: {";
    DT denom = target.data[0] == 0 ? (1 - output.data[0]) : (output.data[0]);
    loss.data[0] = -1.0 / (denom + 0.1);
    if (target.data[0] == 1) {
        std::cout << -log(static_cast<double>(output.data[0] + 0.00001));
    } else {
        std::cout << -log(1.0 - static_cast<double>(output.data[0] + 0.00001));
    }
    std::cout << "}\n";
    std::cout << "loss_data: {" << loss.data[0] << "}\n";

    return loss;
}
    
void Network::backward(const Matrix &output, const Matrix &target) {
    Matrix dE_dy = get_loss(output, target);
    for (int i = layers.size() - 1; i >= 0; --i) {
        bool last = i == 0;
        dE_dy = layers[i].backward(dE_dy, last);
    }
}

void Network::update() {
    for (auto &layer : layers) {
        print("gradient:");
        print(layer.gradient);
        print("bias gradient:");
        print(layer.bias_gradient);
        float lr = -0.01;
        layer.gradient *= lr;
        layer.weights += layer.gradient;
        layer.bias_gradient *= lr;
        layer.bias += layer.bias_gradient;
    }
}