#include <catch2/catch_all.hpp>
#include <cmath>

#include "value.hpp"

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

TEST_CASE("Value self addition", "[arithmetic]") {
    Value a(2.0);
    Value b = a + a;
    REQUIRE(std::abs(b.data() - 4.0) < 1e-6);
}

// Subtraction
TEST_CASE("Value subtraction with positive numbers", "[arithmetic]") {
    Value a(2.0);
    Value b(3.0);
    Value c = a - b;
    REQUIRE(std::abs(c.data() - (-1.0)) < 1e-6);
}

TEST_CASE("Value subtraction with positive and negative numbers", "[arithmetic]") {
    Value d(-1.0);
    Value e(1.0);
    Value f = d - e;
    REQUIRE(std::abs(f.data() - (-2.0)) < 1e-6);
}

TEST_CASE("Value subtraction with zero values", "[arithmetic]") {
    Value g(0.0);
    Value h(0.0);
    Value i = g - h;
    REQUIRE(std::abs(i.data() - 0.0) < 1e-6);
}

TEST_CASE("Value subtraction with decimal numbers", "[arithmetic]") {
    Value j(1.5);
    Value k(2.5);
    Value l = j - k;
    REQUIRE(std::abs(l.data() - (-1.0)) < 1e-6);
}

TEST_CASE("Value subtraction with negative numbers", "[arithmetic]") {
    Value m(-2.5);
    Value n(-2.5);
    Value o = m - n;
    REQUIRE(std::abs(o.data() - 0.0) < 1e-6);
}

TEST_CASE("Value self subtraction", "[arithmetic]") {
    Value a(2.0);
    Value b = a - a;
    REQUIRE(std::abs(b.data() - 0.0) < 1e-6);
}

// Multiplication
TEST_CASE("Value multiplication with positive numbers", "[arithmetic]") {
    Value a(2.0);
    Value b(3.0);
    Value c = a * b;
    REQUIRE(std::abs(c.data() - 6.0) < 1e-6);
}

TEST_CASE("Value multiplication with positive and negative numbers", "[arithmetic]") {
    Value d(-1.0);
    Value e(1.0);
    Value f = d * e;
    REQUIRE(std::abs(f.data() - (-1.0)) < 1e-6);
}

TEST_CASE("Value multiplication with zero values", "[arithmetic]") {
    Value g(0.0);
    Value h(0.0);
    Value i = g * h;
    REQUIRE(std::abs(i.data() - 0.0) < 1e-6);
}

TEST_CASE("Value multiplication with decimal numbers", "[arithmetic]") {
    Value j(1.5);
    Value k(2.5);
    Value l = j * k;
    REQUIRE(std::abs(l.data() - 3.75) < 1e-6);
}

TEST_CASE("Value multiplication with negative numbers", "[arithmetic]") {
    Value m(-2.5);
    Value n(-2.5);
    Value o = m * n;
    REQUIRE(std::abs(o.data() - 6.25) < 1e-6);
}

TEST_CASE("Value self multiplication", "[arithmetic]") {
    Value a(3.0);
    Value b = a * a;
    REQUIRE(std::abs(b.data() - 9.0) < 1e-6);
}

// Division
TEST_CASE("Value division with positive numbers", "[arithmetic]") {
    Value a(6.0);
    Value b(3.0);
    Value c = a / b;
    REQUIRE(std::abs(c.data() - 2.0) < 1e-6);
}

TEST_CASE("Value division with positive and negative numbers", "[arithmetic]") {
    Value d(-6.0);
    Value e(3.0);
    Value f = d / e;
    REQUIRE(std::abs(f.data() - (-2.0)) < 1e-6);
}

TEST_CASE("Value division with decimal numbers", "[arithmetic]") {
    Value g(7.5);
    Value h(2.5);
    Value i = g / h;
    REQUIRE(std::abs(i.data() - 3.0) < 1e-6);
}

TEST_CASE("Value division with negative numbers", "[arithmetic]") {
    Value j(-7.5);
    Value k(-2.5);
    Value l = j / k;
    REQUIRE(std::abs(l.data() - 3.0) < 1e-6);
}

TEST_CASE("Value division by zero", "[arithmetic]") {
    Value m(1.0);
    Value n(0.0);
    REQUIRE_THROWS_AS(m / n, std::runtime_error);
}

TEST_CASE("Value self division", "[arithmetic]") {
    Value a(4.0);
    Value b = a / a;
    REQUIRE(std::abs(b.data() - 1.0) < 1e-6);
}

// Power
TEST_CASE("Value power with positive exponent", "[arithmetic]") {
    Value a(2.0);
    Value b = a.pow(3.0);
    REQUIRE(std::abs(b.data() - 8.0) < 1e-6);
}

TEST_CASE("Value power with zero exponent", "[arithmetic]") {
    Value c(2.0);
    Value d = c.pow(0.0);
    REQUIRE(std::abs(d.data() - 1.0) < 1e-6);
}

TEST_CASE("Value power with one exponent", "[arithmetic]") {
    Value e(2.0);
    Value f = e.pow(1.0);
    REQUIRE(std::abs(f.data() - 2.0) < 1e-6);
}

TEST_CASE("Value power with negative exponent", "[arithmetic]") {
    Value g(2.0);
    Value h = g.pow(-2.0);
    REQUIRE(std::abs(h.data() - 0.25) < 1e-6);
}

TEST_CASE("Value power with decimal exponent", "[arithmetic]") {
    Value i(4.0);
    Value j = i.pow(0.5);
    REQUIRE(std::abs(j.data() - 2.0) < 1e-6);
}

TEST_CASE("Value power with negative base and integer exponent", "[arithmetic]") {
    Value k(-2.0);
    Value l = k.pow(3.0);
    REQUIRE(std::abs(l.data() - (-8.0)) < 1e-6);
}

TEST_CASE("Value power with negative base and even exponent", "[arithmetic]") {
    Value m(-2.0);
    Value n = m.pow(2.0);
    REQUIRE(std::abs(n.data() - 4.0) < 1e-6);
}

TEST_CASE("Value power with negative base and fractional exponent", "[arithmetic]") {
    Value o(-4.0);
    REQUIRE_THROWS_AS(o.pow(0.5), std::runtime_error);
}

TEST_CASE("Value power with zero base and positive exponent", "[arithmetic]") {
    Value o(0.0);
    Value p = o.pow(3.0);
    REQUIRE(std::abs(p.data() - 0.0) < 1e-6);
}

TEST_CASE("Value power with zero base and zero exponent", "[arithmetic]") {
    Value q(0.0);
    REQUIRE_THROWS_AS(q.pow(0.0), std::runtime_error);
}
