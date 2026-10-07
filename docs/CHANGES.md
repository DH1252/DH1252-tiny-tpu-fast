# Changes from the baseline

**Baseline:** [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu) at 04ad692, imported unchanged in the first commit.
**New design:** `rtl/ttf_*.v`, an int8 core for inference only.

The baseline sources (`src/`, `mnist_demo/`, `xor_demo/`, `tiny-tpu-hardened/`) are left as they were. Both designs can therefore be built and measured side by side.

Every change below gives:
- **Baseline:** what the upstream design does, with file and line.
- **Now:** what the new design does instead.
- **Why:** the speed, energy or area reason.
- **Status:** how it was checked.

"Model" means `model/ttf_model.py`: a clock-by-clock Python copy of every register and RAM in `rtl/`. Its self-check compares the copy's results with the layer arithmetic. No HDL simulation or synthesis has been run on the new design yet: the testbench and build scripts are ready for a local run.

## Results so far

All cycle counts come from the model, for the MNIST network (784 → 64 → 10). Accuracy is measured on the 10,000 MNIST test images.

| | Baseline | Now |
|---|---|---|
| MACs per clock, peak | 4 (2 × 2 array) | N² (64 for N = 8, 256 for N = 16) |
| MACs per clock, MNIST, batch 1, N = 8 | — | 7.9 (6,470 clocks per image) |
| MACs per clock, MNIST, batch 8, N = 8 | — | 62.8 of 64 (809 clocks per image) |
| MACs per clock, MNIST, batch 16, N = 16 | — | 246 of 256 (206 clocks per image) |
| Operand width | 16 bit (Q8.8) | 8 bit |
| MNIST test accuracy | 97.35 % (Q8.8, as trained) | 97.27 % (int8); 99.70 % of images get the same class |
| Clock | DE1-SoC: 50 MHz, Fmax 77.8 MHz (checked-in Quartus report); ASIC config: 50 ns | not measured yet |

At batch 1 the array is limited by weight bandwidth. Every weight is used once, so an N × N array does N useful MACs per clock. At batch N or more, each weight is reused N times and the array stays busy.

Baseline cycles per MNIST image have not been measured yet. That is phase 0 of `PLAN.md`.

## The changes

### C1. Inference only

- **Baseline:** training in hardware.
  - Loss: `src/loss_parent.sv`, `src/loss_child.sv`.
  - Leaky ReLU derivative: `src/leaky_relu_derivative_*.sv`.
  - Gradient descent: `src/gradient_descent.sv`, instantiated in `src/unified_buffer.sv`.
  - Transpose reads, and H and Y matrix reads for back-propagation in the buffer.
  - VPU pathways for the backward pass (`src/vpu.sv:5-16`).
- **Now:** a forward pass only. None of the training logic exists, so none of it costs area or switches.
- **Why:** area and energy. You asked for inference only.

### C2. int8 operands, exact integer sums, one rescaling per output

- **Baseline:** 16-bit Q8.8 numbers. Every multiply rounds and saturates (`fxp_mul` → `fxp_zoom`, `src/pe.sv:36`). Every add saturates (`fxp_add`, `src/pe.sv:43`).
- **Now:**
  - The multiply is int8 × int8.
  - Partial sums are exact integers, with no rounding or saturation inside the array.
  - Each output is scaled back to int8 once, in its output lane (`rtl/ttf_post.v`): `y = sat8(rnd(sat18(rnd(acc, s0)) × mult, s1))`, then ReLU.
- **Why:**
  - An 8 × 8 multiplier is about a quarter of a 16 × 16 one in area and energy.
  - Rounding and saturation leave the per-MAC path.
  - Half the bits move through the array and the memories.
- **Status:**
  - `tools/ttf_quant_mnist.py` quantizes the upstream model per layer.
  - MNIST test accuracy goes from 97.35 % to 97.27 %.
  - The model reproduces the stored logits bit for bit.

### C3. Partial-sum width matched to the array

- **Baseline:** 16-bit saturating partial sums.
- **Now:** 16 + log2(N) bits (19 for N = 8). That is exactly enough for N int8 products, so no overflow and no saturation logic are needed.
  - Sums over more than N inputs are added up in the output lanes, in ACC_W bits.
  - ACC_W defaults to 32; it is 24 on the Tang Nano 20K.
  - MNIST needs at most 15 bits.
- **Why:** narrower adders and registers in each of the N² PEs.

### C4. Optional pipeline stage in the PE

- **Baseline:** one clock for multiply, round, saturate, add and saturate.
- **Now:** `PIPE = 1` registers the product, so the PE's path is just multiply, or just add.
  - This adds one clock of latency for the whole array, not one per PE: a PE's product is ready in the same clock as the sum from the PE above it.
  - `PIPE = 0` keeps the single-clock MAC.
- **Why:** clock speed. On the Gowin part the product register is the DSP block's output register.
- **Status:** the model checks both settings.

### C5. Weights loaded in place by a diagonal wave

- **Baseline:**
  - Each PE has a shadow register and an active register (`weight_reg_inactive`, `weight_reg_active`; `src/pe.sv:54,71`).
  - New weights shift down through every PE in the column (`pe_weight_out <= pe_weight_in`, `src/pe.sv:72`), so every PE's register switches on every load clock.
  - A switch signal then copies shadow to active.
- **Now:**
  - One 8-bit weight register per PE.
  - Each column has a weight bus that carries row r's weight while a load token is at row r. The token moves down one row per clock, skewed one clock per column.
  - Each PE is written exactly once per tile, in the clock just before the tile's first input reaches it.
  - The next tile's weights load while the current tile is still running. Tiles follow each other every max(M, N) clocks with no stall, where M is the batch.
- **Why:**
  - Half the weight registers.
  - No weight rippling through N registers.
  - Full array use at batch ≥ N.
- **Status:** the model checks this, and mutation tests confirm that an off-by-one in the token or valid timing makes the check fail.

### C6. Nothing switches when idle

- **Baseline:**
  - A PE clears its outputs to zero on every clock without valid data (`src/pe.sv:83`).
  - A PE clears all of its registers while it is disabled (`src/pe.sv:51`).
  - So idle PEs switch as soon as data stops.
- **Now:**
  - Data registers load only on valid data (clock enables); otherwise they hold.
  - RAM read ports are enabled only when needed, and their outputs hold.
  - Address and flag delay lines are the only registers that shift every clock.
- **Why:** dynamic energy. Only useful work switches.

### C7. Synchronous reset, on control flags only

- **Baseline:** asynchronous reset on every register (`always_ff @(posedge clk or posedge rst)`, 10 blocks in `src/`).
- **Now:** a synchronous reset on the valid flags, load tokens and the sequencer only. Data registers have no reset.
- **Why:**
  - Smaller flops: sky130 `dfxtp` instead of `dfrtp`.
  - No reset tree to every data bit.
  - Lets the Gowin tools pack registers into DSP and RAM blocks.

### C8. Parameterised array size

- **Baseline:** 2 × 2 (`SYSTOLIC_ARRAY_WIDTH = 2`). The ports are hard-coded per lane: `_0` and `_1` (`src/unified_buffer.sv:31-55`).
- **Now:** `N` = 4, 8 or 16, everywhere.
- **Why:** throughput grows with N².

### C9. The unified buffer becomes separate RAMs

- **Baseline:**
  - A 128 × 16 flip-flop array (`src/unified_buffer.sv:59`), cleared word by word on reset (`:210`).
  - Seven sets of read pointers, each with 16-bit pointer, size and time-counter registers, and comparators.
  - Wide read multiplexers.
- **Now:** one-write, one-read RAMs (`rtl/ttf_ram.v`). These infer block RAM on the FPGA and map to SRAM macros on an ASIC.
  - **ACT:** activations, one 8-bit RAM per lane.
  - **WGT:** weights, N/G RAMs of G lanes each.
  - **ACC** and **BIAS:** in each output lane.
  - **DESC:** layer descriptors.
- **Why:**
  - A flip-flop bit costs roughly ten times an SRAM bit in area and leakage.
  - Fixed per-lane addressing replaces the pointer logic.

### C10. Skew by delaying addresses, not data

- **Baseline:** inputs are rotated and staggered by the buffer's read logic, which moves 16-bit data through skew stages.
- **Now:**
  - Lane r of the activations is read r clocks after lane 0, from its own RAM, through a shared address delay line.
  - Column c's weight bus is fed c clocks after column 0's in the same way.
  - Output lane c writes when its column's sum is ready.
  - No N(N−1)/2 triangle of data registers is needed at the input or the output.
  - G > 1 packs G weight lanes into one RAM (better block RAM use) and adds G(G−1)/2 byte registers per group.
- **Why:** fewer registers, and the remaining ones carry addresses that change once per row.

### C11. Output lanes replace the VPU

- **Baseline:** the VPU (`src/vpu.sv`) chains bias, leaky ReLU, loss and leaky ReLU derivative, selected by a 4-bit pathway. Each column has a child module.
- **Now:** one output lane per column (`rtl/ttf_post.v`):
  1. adds the bias on the first k-block;
  2. accumulates the k-blocks in its ACC RAM;
  3. on the last k-block, requantizes (C2), applies ReLU, and writes int8 into the activation RAM of the same lane.

  The next layer reads it from there without moving data.
- **Why:** inference needs only bias, rescale and activation. Leaky ReLU is dropped because the model uses ReLU.

### C12. Accumulation over K inside the core

- **Baseline:** the MNIST demo adds partial sums outside the TPU (`mnist_demo/rtl/mnist_classifier_core.v`, header: "Accumulates raw partial sums externally").
- **Now:** the core handles any K. A layer runs as tiles of N × N weights, and the output lanes accumulate across k-blocks.

### C13. A descriptor sequencer instead of a 130-bit instruction per clock

- **Baseline:** `src/control_unit.sv` only splits a 130-bit instruction that the host has to supply on every clock.
- **Now:**
  - Each layer is one 8-word descriptor in a small RAM: addresses, block counts, batch, bias row, requantization, ReLU.
  - `rtl/ttf_seq.v` runs the whole network from a single start: loops over output blocks, input blocks and rows, then drains between layers.
- **Why:** the host link no longer limits throughput, and the host does nothing while the network runs.

### C14. Host bus and counters

- **Now:** a 32-bit word bus with a memory map (`rtl/ttf_core.v` header):
  - WGT, ACT, BIAS and DESC windows;
  - CTRL and STATUS;
  - CYCLES (clocks of the last run);
  - ID and PARAMS, so the host can check the build.

### C15. Tang Nano 20K as the first hardware target

- **Baseline:** DE1-SoC (Cyclone V) with Quartus, JTAG and serial demos.
- **Now:** `boards/tn20k/`.
  - A UART bridge with ping, write, burst write and read, and a 64-byte reply FIFO.
  - A Gowin build (`ttf_gowin.tcl`) with the core clock from the rPLL (`CLK_MHZ`). Its tool settings aim at speed without spending energy for nothing (C18).
  - The PC side is `tools/ttf_host.py`. It checks every result bit for bit and reports clocks, MACs per clock and images per second.
  - Default fit: N = 8, G = 4, ACC_W = 24, 6,400 weight rows (exactly the MNIST network), batch up to 16. That is an estimated 35 of 46 block RAMs.
- **Status:** not built yet.

### C16. ASIC configuration for speed

- **Baseline:** `tiny-tpu-hardened/openlane/config.json`: 50 ns clock, `SYNTH_STRATEGY` "AREA 0".
- **Now:** `asic/config.json`: 10 ns first target, "DELAY 1", timing repair on.
  - The wrapper `asic/ttf_asic_top.v` keeps the RAMs small, because this first run builds them from flip-flops.
  - SRAM macros are phase 3.
- **Status:** not run yet.

### C17. Verification

- **Baseline:** cocotb tests per module (`test/`), ModelSim scripts for the demos.
- **Now:**
  - **Clock-level model:** `model/ttf_model.py`. It checks N = 4, 8, 16, G = 1, 2, 4, both PIPE settings, random multi-layer networks and the MNIST network.
  - **RTL testbench:** `sim/tb_ttf_core.v`, using images from `tools/ttf_image.py` (`make -f fast.mk sim`). Not run yet.
  - **Elaboration:** every new file is elaborated with slang in several configurations, and passes.

### C18. Gowin tool settings for speed and energy

- **Baseline:** Quartus projects for the DE1-SoC with the tool defaults. The PQSE-style Gowin flow this script started from asked for `-opt_goal speed`, which is not a valid value, so it was skipped.
- **Now:** every option is taken from Gowin's Tcl guide (SUG1220-2.1E) and can be overridden from the environment. Each one goes through `try_option`, so an older gw_sh reports and skips what it doesn't know.

  | Option | Value | Why |
  |---|---|---|
  | `-global_freq` | clock × 1.1 (`SYN_MARGIN`) | synthesis aims a little above the target |
  | `-opt_goal` | timing | timing-driven synthesis |
  | `-map_option` | 3 | LUT5 mapping that may spend LUTs for timing; 4 (LUT5/LUT6) is left out, as it spends more for the last few per cent |
  | `-dsp_style` | dsp | the 8 × 8 products and the requantization multiplies in DSP blocks: faster and far less energy per multiply than LUTs |
  | `-netlist_hierarchy` | 0 | flat netlist, optimization across the PE and lane boundaries |
  | `-rw_check_on_ram` | 0 | no bypass logic: the core never reads and writes one address in the same clock |
  | `-ram_style` | auto | block RAM for the big RAMs, distributed RAM for the small ones; forcing block RAM would waste blocks on the accumulators |
  | `-timing_driven` | 1 | |
  | `-place_option` | 2 | placement with timing first (3 or 4: try harder) |
  | `-route_option` | 1 | better routing at the cost of run time |
  | `-retiming_resource` | all | registers moved across DSP and block RAM boundaries |
  | `-replicate_resources` | 1 | copies of high-fanout drivers (the column weight buses, the run / host multiplexers) |
  | `-correct_hold_violation`, `-set_route_vcc` | 1 | the defaults, stated |
  | `-unused_pin` | default | inputs with a weak pull-up; open drain could pull current through the board's pull-ups |

  The I/O-as-GPIO options (`-use_mspi_as_gpio`, `-use_sspi_as_gpio`) are gone; this design uses none of those pins.
- **Why:** on this design, speed and energy mostly pull the same way. Energy per inference is power × time. The static share shrinks as the clock rises, and the dynamic share depends on what switches (C6), not on the clock.
- **Also:** the end of the run prints the timing report's Fmax, the setup and hold violations, and the resource use.
- **Status:** checked with a Tcl interpreter against stand-ins for the Gowin commands, both as the 2.1E guide documents them and as an older version without the 2.0E/2.1E options. Not run with gw_sh yet.

## Not done yet

- HDL simulation, synthesis, Fmax and power for both designs. That is phase 0 of `PLAN.md`, run locally.
- Clock-gating cells in place of the clock enables (phase 4).
- SRAM macros for the ASIC (phase 3).
- Per-channel requantization. This would need a multiplier per output channel instead of per layer.
