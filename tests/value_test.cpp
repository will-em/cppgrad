#include "value.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_addition() {
    Value a(2.0);
    Value b(3.0);
    Value c = a + b;
    assert(std::abs(c.data() - 5.0) < 1e-6);
}

int main() {
    test_addition();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}