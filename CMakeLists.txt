cmake_minimum_required(VERSION 3.19)
project(FastGaussianPuff)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/CGaussianPuff.cpp
)

set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")

find_package (Eigen3 REQUIRED NO_MODULE)
find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
find_package (pybind11 CONFIG)

if(NOT Python3_SOABI)
  message(FATAL_ERROR "Python3_SOABI is empty. Set it to the output of 'python3-config --extension-suffix'")
endif()

# https://stackoverflow.com/questions/25421479/clang-and-undefined-symbols-when-building-a-library
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()

add_library(CGaussianPuff.${Python3_SOABI} SHARED ${SOURCES})
# https://gitlab.dune-project.org/core/dune-common/-/issues/201
set_target_properties(CGaussianPuff.${Python3_SOABI} PROPERTIES SUFFIX ".so") # force .so suffix for mac
set_target_properties(CGaussianPuff.${Python3_SOABI} PROPERTIES PREFIX "") # remove the lib prefix
target_compile_options(CGaussianPuff.${Python3_SOABI} PRIVATE -fPIC -O3 -Wall)
target_link_libraries(CGaussianPuff.${Python3_SOABI} PRIVATE pybind11::pybind11 Eigen3::Eigen)

install(TARGETS CGaussianPuff.${Python3_SOABI} DESTINATION ${Python3_SITEARCH}/FastGaussianPuff)
install(FILES src/GaussianPuff.py src/__init__.py DESTINATION ${Python3_SITEARCH}/FastGaussianPuff)