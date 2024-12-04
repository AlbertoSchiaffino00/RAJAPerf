# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Alberto Schiaffino
# Used to disassemble the binary

# Buildroot contains the GCC toolchain
BR_OUTPUT_DIR ?= $(HERO_ROOT)/cva6-sdk/buildroot/output
RISCV          = $(BR_OUTPUT_DIR)/host
RV64_SYSROOT   = $(RISCV)/riscv64-buildroot-linux-gnu/sysroot

# Makefile hacks
comma:= ,
empty:=
space:= $(empty) $(empty)

# Binaries
HOST_OBJDUMP := $(RISCV)/bin/riscv64-buildroot-linux-gnu-objdump
DEV_OBJDUMP  := $(HERO_INSTALL)/bin/llvm-objdump
EXE := ./build/bin/raja-perf-omptarget.exe
# Objdump
$(EXE).dis: $(EXE)
	@echo "OBJDUMP <= $<"
	@$(HOST_OBJDUMP) -d $^ > $@


$(EXE).dev.dis: $(EXE)
	@echo "OBJDUMP (device) <= $<"
	@llvm-readelf -S  $(EXE) | grep '.rodata' | awk '{print "echo $$[0x"$$4" - 0x"$$5"]"}' | bash > $<.rodata_off
	@llvm-readelf -S --all $^ | grep '\s\.omp_offloading.device_image\>' \
			| awk '{print "dd if=$^ of=$^_riscv.elf bs=1 count=" $$3 " skip=$$[0x" $$2 " - $$(< $<.rodata_off)]"}' \
			| bash \
			&& $(DEV_OBJDUMP) -S $^_riscv.elf > $@ 

.PHONY: dis 
dis: $(EXE).dis $(EXE).dev.dis


ifndef HERO_INSTALL
$(error HERO_INSTALL is not set)
endif
