# fast.mk - tiny-tpu-fast (the int8 inference core in rtl/). The upstream Makefile is left
# as it was, for the baseline.
#
#   make -f fast.mk model            clock-level model vs. the layer arithmetic (Python)
#   make -f fast.mk quant MNIST=DIR  re-quantize the MNIST model (numpy; MNIST idx files in DIR)
#   make -f fast.mk sim [M=8] [RANDOM=seed] [VERILATOR=1]   RTL testbench (iverilog or Verilator)
#   make -f fast.mk tn20k [CLK_MHZ=100] [BAUD=3000000] [N=8 G=4 ...]   Tang Nano 20K bitstream
#   make -f fast.mk host PORT=/dev/ttyUSB1 [BAUD=...] [CLK_MHZ=...] [MNIST=DIR]  run on the board
#   make -f fast.mk asic             OpenLane run of asic/config.json (sky130)
#
# GW_SH: Gowin EDA's gw_sh (e.g. ~/GOWIN_EDA/IDE/bin/gw_sh). OPENLANE: the openlane command.

PY        ?= python3
BUILD     ?= build
M         ?= 8
N         ?= 8
GW_SH     ?= gw_sh
OPENLANE  ?= openlane
RTL       := $(sort $(wildcard rtl/ttf_*.v))
SIMDIR    := $(BUILD)/sim

.PHONY: model quant image sim tn20k host asic clean-fast

model:
	$(PY) model/ttf_model.py

quant:
	@test -n "$(MNIST)" || { echo "MNIST=<directory with the MNIST idx .gz files>"; exit 1; }
	$(PY) tools/ttf_quant_mnist.py --mnist $(MNIST)

image:
	$(PY) tools/ttf_image.py --m $(M) --n $(N) --out $(SIMDIR) $(if $(RANDOM),--random $(RANDOM))

sim: image
ifeq ($(VERILATOR),1)
	verilator --binary -j 0 -Wno-fatal -Wno-WIDTH --top-module tb_ttf_core -GDIR='"$(SIMDIR)"' \
	    -GN=$(N) -Mdir $(SIMDIR)/obj -o ../vtb $(RTL) sim/tb_ttf_core.v
	$(SIMDIR)/vtb | tee $(SIMDIR)/sim.log
else
	iverilog -g2005 -s tb_ttf_core -Ptb_ttf_core.DIR='"$(SIMDIR)"' -Ptb_ttf_core.N=$(N) \
	    -o $(SIMDIR)/tb.vvp $(RTL) sim/tb_ttf_core.v
	vvp -n $(SIMDIR)/tb.vvp | tee $(SIMDIR)/sim.log
endif
	@grep -q "^PASS" $(SIMDIR)/sim.log

tn20k:
	QT_QPA_PLATFORM=$${QT_QPA_PLATFORM:-offscreen} QT_XCB_GL_INTEGRATION=none LIBGL_ALWAYS_SOFTWARE=1 \
	    CLK_MHZ=$(CLK_MHZ) BAUD=$(BAUD) N=$(if $(filter command line,$(origin N)),$(N)) G=$(G) PIPE=$(PIPE) ACC_W=$(ACC_W) \
	    AW_A=$(AW_A) AW_W=$(AW_W) W_D=$(W_D) AW_M=$(AW_M) AW_B=$(AW_B) AW_D=$(AW_D) GOAL=$(GOAL) STEP=$(STEP) \
	    $(GW_SH) boards/tn20k/ttf_gowin.tcl

host:
	@test -n "$(PORT)" || { echo "PORT=<serial port of the board>"; exit 1; }
	$(PY) tools/ttf_host.py --port $(PORT) $(if $(BAUD),--baud $(BAUD)) $(if $(CLK_MHZ),--mhz $(CLK_MHZ)) \
	    --m $(M) $(if $(MNIST),--mnist $(MNIST) --images $(or $(IMAGES),1000))

asic:
	cd asic && $(OPENLANE) config.json

clean-fast:
	rm -rf $(BUILD)
