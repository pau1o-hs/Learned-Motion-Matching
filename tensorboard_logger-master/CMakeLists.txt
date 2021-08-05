project(tensorboard_logger)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 11)
find_package(Protobuf REQUIRED)

file(GLOB protos "proto/*.proto")

if(NOT PROTOBUF_FOUND)
    message (FATAL_ERROR "Cannot find Protobuf")
endif()

file(GLOB sources "src/*.cc")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${protos})

set(TENSORBOARD_INCLUDE_PATH
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR})

include_directories(
    ${TENSORBOARD_INCLUDE_PATH}
    ${Protobuf_INCLUDE_DIRS}
)

add_library(tensorboard_logger STATIC
  ${sources}
  ${PROTO_SRCS}
)

add_executable(test
    tests/test_tensorboard_logger.cc
)

target_link_libraries(test tensorboard_logger ${Protobuf_LIBRARIES})
