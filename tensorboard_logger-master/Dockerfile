FROM ubuntu:20.04

LABEL description="Standalone TensorBoard logger API in C++" \
      maintainer="i@toonaive.me" \
      version="1.0" \
      url="https://github.com/RustingSword/tensorboard_logger"

ENV DEBIAN_FRONTEND="noninteractive" TZ="Etc/UTC"

RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/RustingSword/tensorboard_logger

