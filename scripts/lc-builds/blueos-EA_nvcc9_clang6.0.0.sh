#!/usr/bin/env bash

##
## Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read RAJAPerf/LICENSE.
##

BUILD_SUFFIX=lc_blueos-EA-nvcc9-clang6.0.0
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/nvcc_clang_6_0_0.cmake

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-9.2.148 \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc \
  -DCUDA_ARCH=sm_60 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
