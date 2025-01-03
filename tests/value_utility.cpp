#include "catch2/catch.hpp"
#include "value.hpp"
#include <sstream>

TEST_CASE("Value string representation", "[utility]") {
    Value v(3.14);
    std::string str = v.str();
    REQUIRE(str.find("data=3.14") != std::string::npos);
    REQUIRE(str.find("grad=0") != std::string::npos);
    REQUIRE(str.find("op=''") != std::string::npos);
}

TEST_CASE("Value stream operator", "[utility]") {
    Value v(3.14);
    std::ostringstream oss;
    oss << v;
    std::string str = oss.str();
    REQUIRE(str == v.str());
}