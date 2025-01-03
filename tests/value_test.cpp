#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "value.hpp"
#include <cmath>

// Addition
TEST_CASE("Value addition with positive numbers", "[arithmetic]") {
    Value a(2.0);
    Value b(3.0);
    Value c = a + b;
    REQUIRE(std::abs(c.data() - 5.0) < 1e-6);
}

TEST_CASE("Value addition with positive and negative numbers", "[arithmetic]") {
    Value d(-1.0);
    Value e(1.0);
    Value f = d + e;
    REQUIRE(std::abs(f.data() - 0.0) < 1e-6);
}

TEST_CASE("Value addition with zero values", "[arithmetic]") {
    Value g(0.0);
    Value h(0.0);
    Value i = g + h;
    REQUIRE(std::abs(i.data() - 0.0) < 1e-6);
}

TEST_CASE("Value addition with decimal numbers", "[arithmetic]") {
    Value j(1.5);
    Value k(2.5);
    Value l = j + k;
    REQUIRE(std::abs(l.data() - 4.0) < 1e-6);
}

TEST_CASE("Value addition with negative numbers", "[arithmetic]") {
    Value m(-2.5);
    Value n(-2.5);
    Value o = m + n;
    REQUIRE(std::abs(o.data() - (-5.0)) < 1e-6);
}
// Gradient Computation
TEST_CASE("Gradient computation for addition with positive numbers", "[gradient]") {
    Value a(2.0);
    Value b(3.0);
    Value c = a + b;
    c.backward();
    REQUIRE(std::abs(a.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(b.grad() - 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for addition with positive and negative numbers", "[gradient]") {
    Value d(-1.0);
    Value e(1.0);
    Value f = d + e;
    f.backward();
    REQUIRE(std::abs(d.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(e.grad() - 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for addition with zero values", "[gradient]") {
    Value g(0.0);
    Value h(0.0);
    Value i = g + h;
    i.backward();
    REQUIRE(std::abs(g.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(h.grad() - 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for addition with decimal numbers", "[gradient]") {
    Value j(1.5);
    Value k(2.5);
    Value l = j + k;
    l.backward();
    REQUIRE(std::abs(j.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(k.grad() - 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for addition with negative numbers", "[gradient]") {
    Value m(-2.5);
    Value n(-2.5);
    Value o = m + n;
    o.backward();
    REQUIRE(std::abs(m.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(n.grad() - 1.0) < 1e-6);
}
