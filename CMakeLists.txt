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

# Add the include directories for the BLAS and LAPACK libraries
include_directories("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers")


# Define the source files and headers
file(GLOB headers "include/*.h")

# Add the library target
add_library(HMatrixCPP INTERFACE ${headers})

# Link the library with the BLAS and LAPACK libraries and set the output name
target_link_libraries(HMatrixCPP INTERFACE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Check if we are on macOS
if(APPLE)
    # Check if the macOS version is greater than or equal to 13.3
    if(CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "21.4.0") # 21.4.0 corresponds to macOS 13.3
        # Add compile definitions for updated LAPACK interface
        add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
    endif()
endif()

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
target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)


if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nullability-extension")
    add_compile_definitions(__APPLE__)
endif()

# Link the test executable with the library and Google Test
target_link_libraries(unit_tests gtest gtest_main ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} HMatrixCPP)

# Add the tests to be run
add_test(NAME unit_test COMMAND unit_tests)

