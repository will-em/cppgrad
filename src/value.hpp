#ifndef CPPGRAD_VALUE_HPP
#define CPPGRAD_VALUE_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

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
    explicit Value(double data, const std::vector<DataPtr> children = {}, const std::string& op = "");
    Value(const Value&) = default;
    Value(Value&& other) noexcept = default;

    Value& operator=(const Value&) = default;
    Value& operator=(Value&& other) noexcept = default;

    // Inline accessors
    double data() const noexcept { return data_ptr->data; }
    double grad() const noexcept { return data_ptr->grad; }
    std::string op() const noexcept { return data_ptr->op; }

    void set_data(double new_data) noexcept { data_ptr->data = new_data; }
    void set_grad(double new_grad) noexcept { data_ptr->grad = new_grad; }

    std::string str() const;

    // Operations
    Value operator+(const Value& other) const;
    Value operator-(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator/(const Value& other) const;
    Value pow(double exponent) const;
    Value relu() const;
    void backward();

    friend std::ostream& operator<<(std::ostream& os, const Value& v);
};

#endif  // CPPGRAD_VALUE_HPP