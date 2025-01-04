#include "neuron.hpp"

#include <numeric>
#include <random>

Neuron::Neuron(size_t input_size, bool use_nonlinearity) : use_nonlinearity_(use_nonlinearity) {
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    weights_.reserve(input_size + 1);  // +1 for bias
    for (size_t i = 0; i < input_size; ++i) {
        weights_.emplace_back(Value(distribution(random_generator)));
    }
    weights_.emplace_back(Value(0.0));  // bias initialized to 0
}

Value Neuron::operator()(const std::vector<Value>& inputs) {
    Value activation = weights_.back();  // bias term
    activation = std::inner_product(weights_.begin(), weights_.end() - 1, inputs.begin(), activation);
    return use_nonlinearity_ ? activation.relu() : activation;
}

std::vector<Value> Neuron::parameters() { return weights_; }

std::string Neuron::str() const {
    return (use_nonlinearity_ ? "ReLU" : "Linear") + std::string("Neuron(") + std::to_string(weights_.size()) + ")";
}

std::ostream& operator<<(std::ostream& os, const Neuron& n) { return os << n.str(); }
