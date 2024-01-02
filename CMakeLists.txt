# Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Define the project name and the language used
project(HMatrixCPP VERSION 1.0 DESCRIPTION "Hierarchical Matrices Library in C++" LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the BLAS and LAPACK libraries
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add the include directories
include_directories(include)

# Define the source files and headers
file(GLOB SOURCES "src/*.cpp")

# Add the library target
add_library(HMatrixCPP STATIC ${SOURCES})

#include google test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

# Enable testing with CTest
enable_testing()

# Define the test executable
file(GLOB TEST_SOURCES "tests/unit_tests/*.cpp")
add_executable(unit_tests ${TEST_SOURCES})

# Link the test executable with the library and Google Test
target_link_libraries(unit_tests HMatrixCPP gtest gtest_main ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Add the tests to be run
add_test(NAME unit_test COMMAND unit_tests)

