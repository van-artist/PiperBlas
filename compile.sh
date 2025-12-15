#!/bin/bash
rm -rf build
mkdir build

rm -rf build
cmake -S . -B build \
  -DPIPER_ENABLE_CUDA=ON \
  -DPIPER_ENABLE_CPU_BLAS=ON -DPIPER_REQUIRE_CPU_BLAS=ON \
  -DPIPER_ENABLE_MKL=ON -DMKL_ROOT="$MKLROOT" \
  -DPIPER_OPENBLAS_LIB=/lib/x86_64-linux-gnu/libopenblas.so
cmake --build build -j
