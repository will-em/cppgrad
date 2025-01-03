#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>

class Value {
private:
    struct Data;
    using DataPtr = std::shared_ptr<Data>;

    struct Data {
        double data;
        double grad;
        std::vector<DataPtr> children;
        std::function<void()> backward_fn;
        std::string op;

        explicit Data(double data, const std::vector<DataPtr>& children = {}, const std::string& op = "")
            : data(data), grad(0.0), children(children), backward_fn([]() {}), op(op) {}
    };

    DataPtr data_ptr;

public:
    // Constructors
    explicit Value(double data, const std::vector<DataPtr> children = {}, const std::string& op = "")
        : data_ptr(std::make_shared<Data>(data, children, op)) {}

    Value(const Value&) = default;
    Value(Value&& other) noexcept = default;

    // Assignment operators
    Value& operator=(const Value&) = default;
    Value& operator=(Value&& other) noexcept = default;

    // Accessors
    double data() const noexcept { return data_ptr->data; }
    double grad() const noexcept { return data_ptr->grad; }
    std::string op() const noexcept { return data_ptr->op; }

    // String representation
    std::string str() const {
        return "Value(data=" + std::to_string(data()) + 
               ", grad=" + std::to_string(grad()) + 
               ", op='" + op() + "')";
    }

    // Operator overloads
    Value operator+(const Value& other) const {
        Value result(data_ptr->data + other.data_ptr->data, {data_ptr, other.data_ptr}, "+");

        result.data_ptr->backward_fn = [result, this, other]() {
            this->data_ptr->grad += result.data_ptr->grad;
            other.data_ptr->grad += result.data_ptr->grad;
        };

        return result;
    }

    Value operator-(const Value& other) const {
        Value result(data_ptr->data - other.data_ptr->data, {data_ptr, other.data_ptr}, "-");

        result.data_ptr->backward_fn = [result, this, other]() {
            this->data_ptr->grad += result.data_ptr->grad;
            other.data_ptr->grad -= result.data_ptr->grad;
        };

        return result;
    }

    Value operator*(const Value& other) const {
        Value result(data_ptr->data * other.data_ptr->data, {data_ptr, other.data_ptr}, "*");

        result.data_ptr->backward_fn = [result, this, other]() {
            this->data_ptr->grad += other.data_ptr->data * result.data_ptr->grad;
            other.data_ptr->grad += this->data_ptr->data * result.data_ptr->grad;
        };

        return result;
    }

    Value operator/(const Value& other) const {
        if (other.data_ptr->data == 0) throw std::runtime_error("Division by zero");
        Value result(data_ptr->data / other.data_ptr->data, {data_ptr, other.data_ptr}, "/");

        result.data_ptr->backward_fn = [result, this, other]() {
            this->data_ptr->grad += result.data_ptr->grad / other.data_ptr->data;
            other.data_ptr->grad -= result.data_ptr->grad * this->data_ptr->data / (other.data_ptr->data * other.data_ptr->data);
        };

        return result;
    }

    // Backward pass
    void backward() {

        // Topological sort
        std::vector<DataPtr> topo_order;
        std::unordered_set<DataPtr> visited;
        std::function<void(const DataPtr&)> buildTopo = [&](const DataPtr& node) {
            if (visited.find(node) != visited.end()) return; // Already visited
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

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        return os << v.str();
    }
};