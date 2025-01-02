cmake_minimum_required(VERSION 3.10)
project(cppgrad VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add main executable
add_executable(cppgrad src/main.cpp)
target_include_directories(cppgrad PRIVATE src)

# Enable testing
enable_testing()

# Add test executable
add_executable(cppgrad_tests tests/value_test.cpp)
target_include_directories(cppgrad_tests PRIVATE src)

# Add test to CTest
add_test(NAME value_test COMMAND cppgrad_tests)