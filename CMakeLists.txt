cmake_minimum_required(VERSION 3.20)
project(einsum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ein ein.cpp)

target_compile_options(ein PRIVATE -Wall -Wextra)
target_include_directories(ein PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME ein COMMAND ein)
