cmake_minimum_required(VERSION 3.14)
project(cppgrad VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add main executable
add_executable(cppgrad src/main.cpp)
target_include_directories(cppgrad PRIVATE src)

# Setup Catch2 using FetchContent
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

# Add test executable
file(GLOB TEST_SOURCES tests/*.cpp)
add_executable(cppgrad_tests ${TEST_SOURCES})
target_include_directories(cppgrad_tests PRIVATE src)
target_link_libraries(cppgrad_tests PRIVATE Catch2::Catch2WithMain)

# Enable testing
enable_testing()

# Register test with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(cppgrad_tests)