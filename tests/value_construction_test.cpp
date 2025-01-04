#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "value.hpp"

TEST_CASE("Value construction and basic properties", "[construction]") {
    Value v(3.14);
    REQUIRE(std::abs(v.data() - 3.14) < 1e-6);
    REQUIRE(std::abs(v.grad() - 0.0) < 1e-6);
    REQUIRE(v.op() == "");
}

TEST_CASE("Value copy construction and assignment", "[construction]") {
    Value v1(3.14);
    Value v2 = v1;  // Copy construction
    Value v3(2.0);
    v3 = v1;  // Copy assignment

    REQUIRE(std::abs(v1.data() - v2.data()) < 1e-6);
    REQUIRE(std::abs(v1.data() - v3.data()) < 1e-6);

    // Modify original - gradients should propagate to copies
    v1.backward();
    REQUIRE(std::abs(v2.grad() - v1.grad()) < 1e-6);  // v2 should be affected
    REQUIRE(std::abs(v3.grad() - v1.grad()) < 1e-6);  // v3 should be affected
}

TEST_CASE("Value move construction and assignment", "[construction]") {
    Value v1(3.14);
    Value v2 = std::move(Value(3.14));  // Move construction
    Value v3(2.0);
    v3 = std::move(Value(3.14));  // Move assignment

    REQUIRE(std::abs(v2.data() - 3.14) < 1e-6);
    REQUIRE(std::abs(v3.data() - 3.14) < 1e-6);
}

TEST_CASE("Value construction with operation info", "[construction]") {
    Value v1(2.0);
    Value v2(3.0);
    Value v3 = v1 + v2;

    REQUIRE(v3.op() == "+");
    REQUIRE(std::abs(v3.data() - 5.0) < 1e-6);
}

TEST_CASE("Value construction with zero", "[construction]") {
    Value v(0.0);
    REQUIRE(std::abs(v.data() - 0.0) < 1e-6);
    REQUIRE(std::abs(v.grad() - 0.0) < 1e-6);
    REQUIRE(v.op() == "");
}

TEST_CASE("Value construction with negative numbers", "[construction]") {
    Value v(-3.14);
    REQUIRE(std::abs(v.data() + 3.14) < 1e-6);
    REQUIRE(std::abs(v.grad() - 0.0) < 1e-6);
    REQUIRE(v.op() == "");
}
