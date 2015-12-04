cmake_minimum_required(VERSION 2.8)
project(embed_julia)

add_definitions(-DJULIA_INIT_DIR=\"/home/kristoffer/Downloads/julia-cbe1bee3a8/lib\")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -std=c++11 -Wall")

set(JULIA_HEADER_DIR /home/kristoffer/Downloads/julia-cbe1bee3a8/include/julia)
set(JULIA_LIB_DIR /home/kristoffer/Downloads/julia-cbe1bee3a8/lib/julia)

include_directories(${JULIA_HEADER_DIR})
# set the path to the library folder
link_directories(${JULIA_LIB_DIR})

# link the libraries to the executable
link_libraries (julia)

set(SOURCE_FILES main.cc ${JULIA_HEADER_DIR}/julia.h JuliaMaterial.cc JuliaMaterial.h)
add_executable(embed_julia ${SOURCE_FILES})


