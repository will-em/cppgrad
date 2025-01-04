#include "neuron.hpp"

#include <catch2/catch_all.hpp>
#include <cmath>

#include "value.hpp"

TEST_CASE("Neuron construction with ReLU", "[neuron]") {
    Neuron n(3);
    auto params = n.parameters();
    REQUIRE(params.size() == 4);           // 3 weights + 1 bias
    REQUIRE(params.back().data() == 0.0);  // bias initialized to 0
}

TEST_CASE("Neuron construction without ReLU", "[neuron]") {
    Neuron n(2, false);
    auto params = n.parameters();
    REQUIRE(params.size() == 3);  // 2 weights + 1 bias
}

TEST_CASE("Linear neuron forward pass", "[neuron]") {
    Neuron n(2, false);
    auto params = n.parameters();
    params[0].set_data(1.0);  // First weight
    params[1].set_data(1.0);  // Second weight
    params[2].set_data(0.0);  // Bias

    std::vector<Value> input = {Value(2.0), Value(3.0)};
    Value output = n(input);
    REQUIRE(std::abs(output.data() - 5.0) < 1e-6);  // 2*1 + 3*1 + 0
}

TEST_CASE("ReLU neuron forward pass with positive output", "[neuron]") {
    Neuron n(2, true);
    auto params = n.parameters();
    params[0].set_data(1.0);
    params[1].set_data(1.0);
    params[2].set_data(0.0);

    std::vector<Value> input = {Value(2.0), Value(3.0)};
    Value output = n(input);
    REQUIRE(std::abs(output.data() - 5.0) < 1e-6);
}

TEST_CASE("ReLU neuron forward pass with negative output", "[neuron]") {
    Neuron n(2, true);
    auto params = n.parameters();
    params[0].set_data(-1.0);
    params[1].set_data(-1.0);
    params[2].set_data(0.0);

    std::vector<Value> input = {Value(2.0), Value(3.0)};
    Value output = n(input);
    REQUIRE(std::abs(output.data() - 0.0) < 1e-6);  // ReLU clamps negative to 0
}

TEST_CASE("Neuron gradient operations", "[neuron]") {
    Neuron n(2);
    auto params = n.parameters();

    // Set some gradients
    for (auto& p : params) {
        p.set_grad(1.0);
    }

    // Verify gradients are set
    for (const auto& p : params) {
        REQUIRE(p.grad() == 1.0);
    }

    // Zero gradients
    n.zero_grad();

    // Verify gradients are zeroed
    for (const auto& p : params) {
        REQUIRE(p.grad() == 0.0);
    }
}

TEST_CASE("ReLU neuron string representation", "[neuron]") {
    Neuron n(3);
    REQUIRE(n.str() == "ReLUNeuron(4)");  // 3 weights + 1 bias
}

TEST_CASE("Linear neuron string representation", "[neuron]") {
    Neuron n(3, false);
    REQUIRE(n.str() == "LinearNeuron(4)");  // 3 weights + 1 bias
}
