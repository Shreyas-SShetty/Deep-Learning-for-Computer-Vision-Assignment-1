cmake_minimum_required(VERSION 3.15)
project(cpp_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
        COMMAND python -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(pybind11_DIR ${PYBIND11_DIR})

find_package(pybind11 REQUIRED)

add_executable(mini_autograd
        layers.cpp
        loss.cpp
        optim.cpp
        stats.cpp
        tensor.cpp
        ops.cpp
        main.cpp
)

pybind11_add_module(cpp_backend_ext
        bindings.cpp
        tensor.cpp
        ops.cpp
        layers.cpp
        loss.cpp
        optim.cpp
        stats.cpp
)
