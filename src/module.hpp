#ifndef CPPGRAD_MODULE_HPP
#define CPPGRAD_MODULE_HPP

#include <vector>

#include "value.hpp"

class Module {
   public:
    virtual ~Module() = default;

    virtual void zero_grad() {
        auto params = parameters();
        for (auto& p : params) {
            p.set_grad(0.0);
        }
    }

    virtual std::vector<Value> parameters() = 0;
};

#endif  // CPPGRAD_MODULE_HPP
