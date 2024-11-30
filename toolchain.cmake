set(CMAKE_SYSTEM_NAME Linux)

# set var for the project
set(HERO_INSTALL_PATH /scratch2/msc24h6/hero-tools/install)
set(RISCV /scratch2/msc24h6/hero-tools/cva6-sdk/buildroot/output/host)
set(RV64_SYSROOT  /scratch2/msc24h6/hero-tools/cva6-sdk/buildroot/output/host/riscv64-buildroot-linux-gnu/sysroot)
set(HERO_ROOT /scratch2/msc24h6/hero-tools)
set(CARFIELD_ROOT  /scratch2/msc24h6/hero-tools/platforms/carfield)
set(RAJAPERF /scratch2/msc24h6/RAJAPerf)

set(CMAKE_FIND_ROOT_PATH  ${RV64_SYSROOT} CACHE PATH "" FORCE)
set(CMAKE_SYSROOT  ${RV64_SYSROOT} CACHE PATH "" FORCE)

set(CMAKE_CXX_COMPILER  ${HERO_INSTALL_PATH}/bin/clang++ CACHE PATH "" FORCE)
set(CMAKE_C_COMPILER    ${HERO_INSTALL_PATH}/bin/clang CACHE PATH "" FORCE)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -g  -debug -v  -save-temps=obj --gcc-toolchain=${RISCV} -target riscv64-hero-linux-gnu")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g  -debug -v  -save-temps=obj --gcc-toolchain=${RISCV} -target riscv64-hero-linux-gnu")

include_directories(${HERO_ROOT}/sw/libhero/include ${HERO_ROOT}/apps/carfield/omp/common )
include_directories(${RV64_SYSROOT}/../include/c++/10.3.0/riscv64-buildroot-linux-gnu)
include_directories(${RV64_SYSROOT}/../include/c++/10.3.0)
include_directories(${RV64_SYSROOT}/usr/include)
include_directories(/scratch2/msc24h6/RAJAPerf/src/OMPTarget_offloads/include)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH )
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH )
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY )


set(CMAKE_LIBRARY_PATH "${HERO_ROOT}/sw/libhero/lib ${HERO_ROOT}/sw/libomp/lib ${RV64_SYSROOT}/usr/include" CACHE PATH "" FORCE)
set(CMAKE_INCLUDE_PATH "${HERO_ROOT}/sw/libhero/include" CACHE PATH "" FORCE)

set(CMAKE_PREFIX_PATH ${RV64_SYSROOT}/usr CACHE PATH "" FORCE)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)


# link
add_link_options(--ld-path=${RISCV}/bin/riscv64-buildroot-linux-gnu-ld -L${HERO_ROOT}/sw/libhero/lib -L${HERO_ROOT}/sw/libomp/lib)
add_link_options(-lm -lhero_spatz_cluster)   

#openmp flags
set(BLT_OPENMP_COMPILE_FLAGS "-fopenmp=libomp;-fopenmp-targets=riscv32-hero-hero1-elf;--hero1-sysroot=${HERO_INSTALL_PATH}/rv32imafdvzfh-ilp32d/riscv32-unknown-elf;-hero1-march=rv32imafdvzfh_xdma;-hero1-D__HERO_1;-hero1-D__HERO_DEV;-hero1-I${CARFIELD_ROOT}/spatz/sw/snRuntime/include;-hero1-I${CARFIELD_ROOT}/spatz/sw/snRuntime/vendor;-hero1-I${CARFIELD_ROOT}/spatz/sw/snRuntime/vendor/riscv-opcodes;-hero1-I${CARFIELD_ROOT}/spatz/sw/spatzBenchmarks/omptarget;-hero1-I${RAJAPERF}/src/basic;-hero1-I${RAJAPERF}/src;-hero1-I/scratch2/msc24h6/hero-tools/apps/carfield/omp/common")
set(BLT_OPENMP_LINK_FLAGS "${BLT_OPENMP_COMPILE_FLAGS};-MF;.deps/.d;/scratch2/msc24h6/RAJAPerf/build/src/basic/CMakeFiles/basic.dir/DAXPY-OMPTarget-out.ll;-lhero_spatz_cluster;-fopenmp=libomp;-fopenmp-targets=riscv32-hero-hero1-elf;-hero1-L${HERO_INSTALL_PATH}/lib/clang/15.0.0/rv32imafdvzfh-ilp32d/lib/;-hero1-lclang_rt.builtins-riscv32;-hero1-T${CARFIELD_ROOT}/spatz/hw/system/spatz_cluster/sw/build/snRuntime/common.ld;-hero1-L${CARFIELD_ROOT}/spatz/hw/system/spatz_cluster/sw/build/spatzBenchmarks;-hero1-lomptarget;-hero1-L${CARFIELD_ROOT}/spatz/hw/system/spatz_cluster/sw/build/snRuntime;-hero1-lsnRuntime-cluster;--hero1-ld-path=${HERO_INSTALL_PATH}/bin/ld.lld")
