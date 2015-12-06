cmake_minimum_required(VERSION 2.8)
project(embed_julia)

add_definitions(-DDEBUG="true")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -std=c++11  -Wall")

set(JULIA_PATH /home/kristoffer/julia0.4)

set(JULIA_HEADER_DIR ${JULIA_PATH}/src)
set(JULIA_LIB_DIR ${JULIA_PATH}/usr/lib)

add_definitions(-DJULIA_INIT_DIR="${JULIA_LIB_DIR}")
include_directories(${JULIA_HEADER_DIR} ${JULIA_HEADER_DIR}/support ${JULIA_PATH}/usr/include )
# set the path to the library folder
link_directories(${JULIA_LIB_DIR})

# link the libraries to the executable
link_libraries (julia)

set(SOURCE_FILES main.cc ${JULIA_HEADER_DIR}/julia.h JuliaMaterialWrapper.cc JuliaMaterialWrapper.h
                 vonMisesNonLinVisc.cc vonMisesNonLinVisc.h)
add_executable(embed_julia ${SOURCE_FILES})


