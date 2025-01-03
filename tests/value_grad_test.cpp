#include <catch2/catch.hpp>
#include "value.hpp"
#include <cmath>

// Addition
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

// Subtraction
TEST_CASE("Gradient computation for subtraction with positive numbers", "[gradient]") {
    Value a(5.0);
    Value b(3.0);
    Value c = a - b;
    c.backward();
    REQUIRE(std::abs(a.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(b.grad() + 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for subtraction with positive and negative numbers", "[gradient]") {
    Value d(1.0);
    Value e(-1.0);
    Value f = d - e;
    f.backward();
    REQUIRE(std::abs(d.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(e.grad() + 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for subtraction with zero values", "[gradient]") {
    Value g(0.0);
    Value h(0.0);
    Value i = g - h;
    i.backward();
    REQUIRE(std::abs(g.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(h.grad() + 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for subtraction with decimal numbers", "[gradient]") {
    Value j(2.5);
    Value k(1.5);
    Value l = j - k;
    l.backward();
    REQUIRE(std::abs(j.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(k.grad() + 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for subtraction with negative numbers", "[gradient]") {
    Value m(-2.5);
    Value n(-1.5);
    Value o = m - n;
    o.backward();
    REQUIRE(std::abs(m.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(n.grad() + 1.0) < 1e-6);
}

// Multiplication
TEST_CASE("Gradient computation for multiplication with positive numbers", "[gradient]") {
    Value a(2.0);
    Value b(3.0);
    Value c = a * b;
    c.backward();
    REQUIRE(std::abs(a.grad() - 3.0) < 1e-6);
    REQUIRE(std::abs(b.grad() - 2.0) < 1e-6);
}

TEST_CASE("Gradient computation for multiplication with positive and negative numbers", "[gradient]") {
    Value d(-1.0);
    Value e(1.0);
    Value f = d * e;
    f.backward();
    REQUIRE(std::abs(d.grad() - 1.0) < 1e-6);
    REQUIRE(std::abs(e.grad() + 1.0) < 1e-6);
}

TEST_CASE("Gradient computation for multiplication with zero values", "[gradient]") {
    Value g(0.0);
    Value h(2.0);
    Value i = g * h;
    i.backward();
    REQUIRE(std::abs(g.grad() - 2.0) < 1e-6);
    REQUIRE(std::abs(h.grad() - 0.0) < 1e-6);
}

TEST_CASE("Gradient computation for multiplication with decimal numbers", "[gradient]") {
    Value j(1.5);
    Value k(2.5);
    Value l = j * k;
    l.backward();
    REQUIRE(std::abs(j.grad() - 2.5) < 1e-6);
    REQUIRE(std::abs(k.grad() - 1.5) < 1e-6);
}

TEST_CASE("Gradient computation for multiplication with negative numbers", "[gradient]") {
    Value m(-2.5);
    Value n(-2.5);
    Value o = m * n;
    o.backward();
    REQUIRE(std::abs(m.grad() + 2.5) < 1e-6);
    REQUIRE(std::abs(n.grad() + 2.5) < 1e-6);
}

// Division
TEST_CASE("Gradient computation for division with positive numbers", "[gradient]") {
    Value a(6.0);
    Value b(3.0);
    Value c = a / b;
    c.backward();
    REQUIRE(std::abs(a.grad() - (1.0 / b.data())) < 1e-6);
    REQUIRE(std::abs(b.grad() + (a.data() / (b.data() * b.data()))) < 1e-6);
}

TEST_CASE("Gradient computation for division with positive and negative numbers", "[gradient]") {
    Value d(-6.0);
    Value e(3.0);
    Value f = d / e;
    f.backward();
    REQUIRE(std::abs(d.grad() - (1.0 / e.data())) < 1e-6);
    REQUIRE(std::abs(e.grad() + (d.data() / (e.data() * e.data()))) < 1e-6);
}

TEST_CASE("Gradient computation for division with zero numerator", "[gradient]") {
    Value g(0.0);
    Value h(2.0);
    Value i = g / h;
    i.backward();
    REQUIRE(std::abs(g.grad() - (1.0 / h.data())) < 1e-6);
    REQUIRE(std::abs(h.grad() + (g.data() / (h.data() * h.data()))) < 1e-6);
}

TEST_CASE("Gradient computation for division with decimal numbers", "[gradient]") {
    Value j(2.5);
    Value k(1.5);
    Value l = j / k;
    l.backward();
    REQUIRE(std::abs(j.grad() - (1.0 / k.data())) < 1e-6);
    REQUIRE(std::abs(k.grad() + (j.data() / (k.data() * k.data()))) < 1e-6);
}

TEST_CASE("Gradient computation for division with negative numbers", "[gradient]") {
    Value m(-2.5);
    Value n(-1.5);
    Value o = m / n;
    o.backward();
    REQUIRE(std::abs(m.grad() - (1.0 / n.data())) < 1e-6);
    REQUIRE(std::abs(n.grad() + (m.data() / (n.data() * n.data()))) < 1e-6);
}