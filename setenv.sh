# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Cyril Koenig <cykoenig@iis.ee.ethz.ch>

# These env variables won't be needed if moving to a full Makefile solutions
# but at the moment they may be usefull to compile code in other repositories
# using the Hero toolchain. They are also used to localize Linux images for
# flashing FPGAs in other repositories

#export CC=/usr/pack/gcc-9.2.0-af/linux-x64/bin/gcc
#export CXX=/usr/pack/gcc-9.2.0-af/linux-x64/bin/g++

export HERO_ROOT=$(pwd)/../../hero-tools
export CVA6_SDK=$HERO_ROOT/cva6-sdk
export HERO_INSTALL=$HERO_ROOT/install
export RISCV=$CVA6_SDK/buildroot/output/host/
export RV64_SYSROOT=$RISCV/riscv64-buildroot-linux-gnu/sysroot
export CROSS_COMPILE=$RISCV/bin/riscv64-buildroot-linux-gnu-
# Parse the device Linux path from buildroot
if [ -f $CVA6_SDK/buildroot/Makefile ]; then
export BR_LINUX_DIR=$CVA6_SDK/buildroot/output/$(make -C $HERO_ROOT/cva6-sdk/buildroot linux-show-info \
                    | python3 -c "import sys, json; print(json.loads(sys.stdin.readlines()[0])['linux']['build_dir'])")
fi
# For Occamy bootrom
export UBOOT_SPL_BIN=$CVA6_SDK/install64/u-boot-spl.bin
export UBOOT_ITB=$CVA6_SDK/install64/u-boot.itb

# Add linux-gcc and clang to path
export PATH=$RISCV/bin:$PATH
export PATH=$HERO_INSTALL/bin:$PATH

export HERO_ARTIFACTS_ROOT=/usr/scratch2/wuerzburg/cykoenig/memora/hero

export CARFIELD_ROOT=$HERO_ROOT/platforms/carfield
export OCCAMY_ROOT=$HERO_ROOT/platforms/occamy

export CC=gcc-7.2.0
export LD_LIBRARY_PATH=/usr/pack/gcc-7.2.0-af/linux-x64/lib/gcc/x86_64-pc-linux-gnu/7.2.0/../../../../lib64:$LD_LIBRARY_PATH
export CXX=g++-7.2.0
# export CPLUS_INCLUDE_PATH=/scratch2/msc24h6/hero-tools/cva6-sdk/buildroot/output/host/riscv64-buildroot-linux-gnu/include/c++/10.3.0/riscv64-buildroot-linux-gnu:/scratch2/msc24h6/hero-tools/cva6-sdk/buildroot/output/host/riscv64-buildroot-linux-gnu/include/c++/10.3.0:$CPLUS_INCLUDE_PATH