cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Compressor)

set(Torch_DIR /home/pau1o-hs/Documents/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB protos "tensorboard_logger-master/proto/*.proto")
file(GLOB sources "tensorboard_logger-master/src/*.cc")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${protos})

set(TENSORBOARD_INCLUDE_PATH
  ${PROJECT_SOURCE_DIR}/tensorboard_logger-master/include
  ${PROJECT_BINARY_DIR})

include_directories(
    ${TENSORBOARD_INCLUDE_PATH}
    ${Protobuf_INCLUDE_DIRS}
)

add_library(tensorboard_logger STATIC
  ${sources}
  ${PROTO_SRCS}
)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp CustomLib.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" tensorboard_logger ${PROTOBUF_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)