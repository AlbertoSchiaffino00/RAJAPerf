# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Cyril Koenig <cykoenig@iis.ee.ethz.ch>

# Buildroot contains the GCC toolchain
BR_OUTPUT_DIR ?= $(realpath $(HERO_ROOT)/cva6-sdk/buildroot/output/)
RISCV          = $(BR_OUTPUT_DIR)/host
RV64_SYSROOT   = $(RISCV)/riscv64-buildroot-linux-gnu/sysroot
RAJAPERF       = $(realpath ../../)
# Makefile hacks
comma:= ,
empty:=
space:= $(empty) $(empty)

# Binaries
LLVM_INSTALL := $(HERO_INSTALL)
CC   := $(HERO_INSTALL)/bin/clang++
LINK := $(HERO_INSTALL)/bin/ld.lld
COB  := $(HERO_INSTALL)/bin/clang-offload-bundler
DIS  := $(HERO_INSTALL)/bin/llvm-dis
HOP  := $(HERO_INSTALL)/bin/hc-omp-pass
GCC  := $(HERO_INSTALL)/bin/$(TARGET_HOST)-gcc
HOST_OBJDUMP := $(RISCV)/bin/riscv64-buildroot-linux-gnu-objdump
DEV_OBJDUMP  := $(HERO_INSTALL)/bin/llvm-objdump
AR   := /scratch2/msc24h6/hero-tools/install/bin/llvm-ar 

# Flags
COB_TARGETS = $(subst $(space),$(comma),host-$(TARGET_HOST) $(foreach target,$(TARGET_DEVS),openmp-$(target)))

# Toolchain(s) selection
CFLAGS   += --gcc-toolchain=$(RISCV) --sysroot=$(RV64_SYSROOT)
CFLAGS   += -target $(TARGET_HOST) 
CFLAGS   += -fopenmp=libomp -fopenmp-targets=$(subst $(space),$(comma),$(foreach target,$(TARGET_DEVS),$(target)))
# Include files used by the OpenMP target RTL
CFLAGS   += -I$(HERO_ROOT)/sw/libhero/include
CFLAGS   += -I$(HERO_ROOT)/apps/carfield/omp/common
CFLAGS   += -I$(RV64_SYSROOT)/../include/c++/10.3.0/riscv64-buildroot-linux-gnu
CFLAGS   += -I$(RV64_SYSROOT)/../include/c++/10.3.0
CFLAGS   += -I$(RAJAPERF)/build/include
CFLAGS   += -I$(RAJAPERF)/tpl/RAJA/include
CFLAGS   += -I$(RAJAPERF)/build/tpl/RAJA/include
CFLAGS   += -I$(RAJAPERF)/tpl/RAJA/tpl/camp/include
CFLAGS   += -I$(RAJAPERF)/build/tpl/RAJA/tpl/camp/include
CFLAGS   += -I$(RAJAPERF)/src

CFLAGS   += -I./include
CFLAGS   += -I./kernels

# Dependancy managements
DEPDIR   := .deps
CFLAGS   += -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d -DCMAKE_BUILD_TYPE="debug"

# Link flags
LDFLAGS  += --ld-path=$(RISCV)/bin/riscv64-buildroot-linux-gnu-ld 
# Path to the OpenMP target RTL
LDFLAGS += -L$(RV64_SYSROOT)/usr/lib
LDFLAGS  += -L$(ROOT)/sw/libomp/lib

APP = $(shell basename `pwd`)
EXE = $(APP)
# Objects for each host/devices
COBJS_UNBUNDLED = $(foreach dev,host $(DEVS),$(patsubst %.cpp, %-$(dev).ll, $(CSRCS)))
# Unique object after bundling host/devices together
COBJS_BUNDLED = $(patsubst %.cpp, %-out.ll, $(CSRCS))

# Targets
all: $(DEPS) $(EXE) libOMPTarget_offloads.ll

# Compile heterogeneous C source and get a bundled .ll
%.ll: %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo "CC     <= $<"
	$(CC) $(debug) -c -emit-llvm -S $(DEPFLAGS) $(CFLAGS) $<

# De-bundle %-host.ll and %-heroX.ll
# Note: We need to replace spaces by comma in COB_OUTPUTS
%-host.ll: %.ll
	@echo "COB    <= $<"
	@COB_OUTPUTS="$(foreach tgt,host $(DEVS),$(<:.ll=-$(tgt).ll))"; \
	COB_CMD="$(COB) -inputs=$< -outputs=\"$${COB_OUTPUTS// /,}\" -type=ll -targets=\"$(COB_TARGETS)\" -unbundle" ; \
	echo $$COB_CMD; \
	eval $$COB_CMD

# Create dependance to %-host.ll for all %-heroX.ll (all created by the rule above)
define add_host_dep =
$(foreach tgt-dev,$(DEVS),$(patsubst %-host.ll, %-$(tgt-dev).ll, $(1))): $(1)
endef
# Call add_cob_dep for all %-host.ll objects
$(foreach host-obj, $(patsubst %.cpp, %-host.ll, $(CSRCS)), $(eval $(call add_host_dep,$(host-obj))))

# Different custom LLVMs passes to be applied on host regions
%-host.OMP.ll: %-host.ll
	echo "there"
	@echo "HOP    <= $<"
	@LLVM_INSTALL=$(HERO_INSTALL)/ $(HOP) $(<) OmpKernelWrapper "HERCULES-omp-kernel-wrapper" $(@:.OMP.ll=.TMP.1.ll)
	@cp $(@:.OMP.ll=.TMP.1.ll) $@

# Different custom LLVMs passes to be applied on devices regions
%.OMP.ll: %.ll
	echo "here"
	@echo "HOP    <= $<"
	@LLVM_INSTALL=$(HERO_INSTALL) $(HOP) $(<) OmpKernelWrapper "HERCULES-omp-kernel-wrapper" $(@:.OMP.ll=.TMP.1.ll)
	@LLVM_INSTALL=$(HERO_INSTALL) $(HOP) $(@:.OMP.ll=.TMP.1.ll) OmpHostPointerLegalizer "HERCULES-omp-host-pointer-legalizer" $(@:.OMP.ll=.TMP.2.ll)
	@cp $(@:.OMP.ll=.TMP.2.ll) $@

# Use COB to re-gather all the targets.OMP.ll into a unique output
%-out.ll: $(foreach dev,host $(DEVS),%-$(dev).OMP.ll)
	@echo "COB    <= $<"
	@COB_INPUTS="$(foreach dev,host $(DEVS),$(<:-host.OMP.ll=-$(dev).OMP.ll))"; \
	COB_CMD="$(COB) -inputs=\"$${COB_INPUTS// /,}\" -outputs=$@ -type=ll -targets=\"$(COB_TARGETS)\""; \
	echo $$COB_CMD; \
	eval $$COB_CMD

# Link the final application
$(EXE): $(COBJS_BUNDLED)


# Create the final library
libOMPTarget_offloads.ll: $(COBJS_BUNDLED)
	@echo "AR     <= $@"
	$(AR) qc $@ $^
	/scratch2/msc24h6/hero-tools/install/bin/llvm-ranlib $@

# Dep
$(DEPDIR):
	@mkdir -p $@

DEPFILES := $(CSRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))

# Phony
clean:
	-rm -vf __hmpp* $(EXE) *~ *.bc *.dis *.elf *.i *.lh *.lk *.ll *.o *.s *.slm a.out* *.dump *.a
	-rm -rvf $(DEPDIR)
	-rm -vf *-host-llvm *-host-gnu

.PHONY: upload
upload: $(EXE)
	[ "${DEPLOY_FILES}" ] && scp ${DEPLOY_FILES} root@hero-vcu128-02.ee.ethz.ch:/root || echo "Sending only binary..."
	scp $? root@172.31.182.94:/root

.PHONY: install
install: $(EXE)
	cp $? $(ROOT)/board/common/overlay/root

.PHONY: dis 
dis: $(EXE).dis $(EXE).dev.dis


ifndef HERO_INSTALL
$(error HERO_INSTALL is not set)
endif
