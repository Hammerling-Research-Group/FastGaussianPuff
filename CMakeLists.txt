cmake_minimum_required(VERSION 3.19)
project(FastGaussianPuff)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    CGaussianPuff.cpp
)

set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")

find_package (Eigen3 REQUIRED NO_MODULE)
find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
find_package (pybind11 CONFIG)

# find_package(LAPACK REQUIRED)
# find_package(LAPACKE REQUIRED)
# find_package(BLAS REQUIRED)

if(NOT Python3_SOABI)
  message(FATAL_ERROR "Python3_SOABI is empty. Set it to the output of 'python3-config --extension-suffix'")
endif()

# message(${Python3_SOABI})

add_library(CGaussianPuff.${Python3_SOABI} SHARED CGaussianPuff.cpp)
set_target_properties(CGaussianPuff.${Python3_SOABI} PROPERTIES PREFIX "") # remove the lib prefix
# add_library(c_fibers_obj MODULE ${SOURCES})
target_compile_options(CGaussianPuff.${Python3_SOABI} PRIVATE -fPIC -O3 -Wall)
target_link_libraries(CGaussianPuff.${Python3_SOABI} PRIVATE pybind11::pybind11 Eigen3::Eigen)

install(TARGETS CGaussianPuff.${Python3_SOABI} DESTINATION ${Python3_SITEARCH}/FastGaussianPuff)

# Set the output name for the shared library
#set_target_properties(c_fibers_obj PROPERTIES OUTPUT_NAME c_fibers_obj)
