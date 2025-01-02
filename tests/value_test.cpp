#include "value.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

void test_addition() {
    Value a(2.0);
    Value b(3.0);
    Value c = a + b;
    assert(std::abs(c.data() - 5.0) < 1e-6);

    Value d(-1.0);
    Value e(1.0);
    Value f = d + e;
    assert(std::abs(f.data() - 0.0) < 1e-6);

    Value g(0.0);
    Value h(0.0);
    Value i = g + h;
    assert(std::abs(i.data() - 0.0) < 1e-6);

    Value j(1.5);
    Value k(2.5);
    Value l = j + k;
    assert(std::abs(l.data() - 3.0) < 1e-6);

    Value m(-2.5);
    Value n(-2.5);
    Value o = m + n;
    assert(std::abs(o.data() - (-5.0)) < 1e-6);
}

int main() {
    test_addition();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}