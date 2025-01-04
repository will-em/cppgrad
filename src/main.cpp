#include <iostream>

#include "value.hpp"

int main() {
    Value a(2.0);
    Value b(3.0);
    Value c = a + b;
    Value d = a + b;
    std::cout << d.data() << std::endl;
    return 0;
}