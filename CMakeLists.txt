cmake_minimum_required(VERSION 3.10)
project(cppgrad VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add main executable
add_executable(cppgrad src/main.cpp)
target_include_directories(cppgrad PRIVATE src)

# Find Catch2 package
find_package(Catch2 REQUIRED)

# Add test executable
add_executable(cppgrad_tests tests/value_test.cpp)
target_include_directories(cppgrad_tests PRIVATE src)
target_link_libraries(cppgrad_tests PRIVATE Catch2::Catch2)

# Enable testing
enable_testing()

# Register test with CTest
include(CTest)
include(Catch)
catch_discover_tests(cppgrad_tests)