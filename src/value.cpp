#include "value.hpp"

#include <cmath>
#include <iostream>
#include <unordered_set>

Value::Value(double data, const std::vector<DataPtr> children, const std::string& op)
    : data_ptr(std::make_shared<Data>(data, children, op)) {}

std::string Value::str() const {
    return "Value(data=" + std::to_string(data()) + ", grad=" + std::to_string(grad()) + ", op='" + op() + "')";
}

Value Value::operator+(const Value& other) const {
    Value result(data_ptr->data + other.data_ptr->data, {data_ptr, other.data_ptr}, "+");

    result.data_ptr->backward_fn = [result, this, other]() {
        this->data_ptr->grad += result.data_ptr->grad;
        other.data_ptr->grad += result.data_ptr->grad;
    };

    return result;
}

Value Value::operator-(const Value& other) const {
    Value result(data_ptr->data - other.data_ptr->data, {data_ptr, other.data_ptr}, "-");

    result.data_ptr->backward_fn = [result, this, other]() {
        this->data_ptr->grad += result.data_ptr->grad;
        other.data_ptr->grad -= result.data_ptr->grad;
    };

    return result;
}

Value Value::operator*(const Value& other) const {
    Value result(data_ptr->data * other.data_ptr->data, {data_ptr, other.data_ptr}, "*");

    result.data_ptr->backward_fn = [result, this, other]() {
        this->data_ptr->grad += other.data_ptr->data * result.data_ptr->grad;
        other.data_ptr->grad += this->data_ptr->data * result.data_ptr->grad;
    };

    return result;
}

Value Value::operator/(const Value& other) const {
    if (other.data_ptr->data == 0) {
        throw std::runtime_error("Division by zero");
    }
    Value result(data_ptr->data / other.data_ptr->data, {data_ptr, other.data_ptr}, "/");

    result.data_ptr->backward_fn = [result, this, other]() {
        this->data_ptr->grad += result.data_ptr->grad / other.data_ptr->data;
        other.data_ptr->grad -=
            result.data_ptr->grad * this->data_ptr->data / (other.data_ptr->data * other.data_ptr->data);
    };

    return result;
}

Value Value::pow(double exponent) const {
    if (data_ptr->data < 0 && std::floor(exponent) != exponent) {
        throw std::runtime_error("Imaginary result not allowed");
    }
    if (data_ptr->data == 0 && exponent <= 0) {
        throw std::runtime_error("Invalid exponentiation");
    }

    Value result(std::pow(data_ptr->data, exponent), {data_ptr}, "pow");

    result.data_ptr->backward_fn = [result, this, exponent]() {
        this->data_ptr->grad += exponent * std::pow(this->data_ptr->data, exponent - 1) * result.data_ptr->grad;
    };

    return result;
}

Value Value::relu() const {
    Value result(std::max(data_ptr->data, 0.0), {data_ptr}, "ReLU");

    result.data_ptr->backward_fn = [result, this]() {
        this->data_ptr->grad += (this->data_ptr->data > 0) ? result.data_ptr->grad : 0.0;
    };

    return result;
}

void Value::backward() {
    std::vector<DataPtr> topo_order;
    std::unordered_set<DataPtr> visited;
    std::function<void(const DataPtr&)> buildTopo = [&](const DataPtr& node) {
        if (visited.find(node) != visited.end()) return;
        visited.insert(node);
        for (const auto& child : node->children) {
            buildTopo(child);
        }
        topo_order.push_back(node);
    };

    buildTopo(data_ptr);

    data_ptr->grad = 1.0;
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        (*it)->backward_fn();
    }
}

std::ostream& operator<<(std::ostream& os, const Value& v) { return os << v.str(); }
