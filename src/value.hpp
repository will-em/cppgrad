#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>


class Value {
private:
    struct Data;
    typedef std::shared_ptr<Data> DataPtr;

    struct Data {
        double data;
        double grad;
        std::vector<DataPtr> children; 
        std::function<void()> backward_fn;           
        std::string op;                          

        Data(double data, const std::vector<DataPtr>& children = {}, const std::string& op = "")
            : data(data), grad(0.0), children(children), backward_fn([]() {}), op(op) {}
    };

    DataPtr data_ptr;

public:
    // Constructor
    Value(double data, const std::vector<DataPtr> children = {}, const std::string& op = "")
        : data_ptr(std::make_shared<Data>(data, children, op)) {}

    // Accessors
    double data() const { return data_ptr->data; }
    double grad() const { return data_ptr->grad; }

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
};