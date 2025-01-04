#ifndef CPPGRAD_NEURON_HPP
#define CPPGRAD_NEURON_HPP

#include <random>
#include <vector>

#include "module.hpp"
#include "value.hpp"

class Neuron : public Module {
   private:
    std::vector<Value> weights_;  // Last weight is bias
    bool use_nonlinearity_;

   public:
    Neuron(size_t input_size, bool use_nonlinearity = true);

    Value operator()(const std::vector<Value>& x);
    std::vector<Value> parameters() override;

    std::string str() const;
    friend std::ostream& operator<<(std::ostream& os, const Neuron& n);
};

#endif  // CPPGRAD_NEURON_HPP
