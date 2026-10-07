# tiny-tpu-fast: plan

A private fork of [tiny-tpu-v2/tiny-tpu](https://github.com/tiny-tpu-v2/tiny-tpu)
(imported at 04ad692) aimed at two numbers:

- **throughput**: MAC/s = (array PEs) × (clock), at a useful utilisation;
- **energy per MAC**: pJ/MAC on a real workload (the MNIST and XOR demos).

The final target is an ASIC (sky130, OpenLane). The Tang Nano 20K (GW2AR-18) is the
first target for testing on hardware: 20,736 LUT4, 48 multipliers of 18 × 18 (each
also two of 9 × 9), 46 block RAMs of 18 Kbit.

Upstream has no licence: nothing here is published until it does (see the first
commit).

## The upstream design, as found

| Where | What | Cost |
|---|---|---|
| `src/pe.sv` | MAC in one cycle: 16 × 16 multiply, round and saturate (`fxp_zoom`), add, saturate | the whole critical path in every PE |
| `src/pe.sv` | partial sums saturate at 16 bits (Q8.8) after every add | accuracy, and two saturation stages per PE |
| `src/pe.sv` | outputs cleared to 0 when not valid; all registers reset while `!pe_enabled` | registers toggle on idle cycles: dynamic power for nothing |
| everywhere | asynchronous reset on datapath registers | larger flops in sky130 (dfrtp vs dfxtp), reset tree |
| `src/systolic.sv` | 2 × 2 array (`SYSTOLIC_ARRAY_WIDTH = 2`) | 4 MACs per clock |
| `src/unified_buffer.sv` | 128 × 16 flip-flop array, 7 read pointers each with 16-bit counters and comparators, wide read multiplexers, a reset loop over every word | flops cost about 10 × the area and power of SRAM; no block RAM inference on FPGA |
| `src/control_unit.sv` | a 130-bit instruction driven by the host every cycle, no instruction memory | the host link limits throughput |
| `tiny-tpu-hardened/openlane/config.json` | `CLOCK_PERIOD` 50 ns, `SYNTH_STRATEGY` AREA 0 | 20 MHz |
| `src/fixedpoint.sv` | a third-party fixed-point library | its own licence to check |

## Phases

Each phase ends with the same measurements, so every change is judged by numbers.
Simulation and synthesis run locally, not in the cloud sessions.

### 0. Baseline

- Scripts that report, for the unchanged design:
  - sky130: Yosys + OpenSTA area, fmax, and power with switching activity from a VCD
    of the MNIST forward pass;
  - Tang Nano 20K: Gowin synthesis and PnR (LUTs, DSPs, BSRAMs, fmax), power estimate.
- A Tang Nano 20K top: a UART bridge to the TPU's instruction and data ports, with a
  PC script that runs the XOR and MNIST demos and checks the results against the
  Python model.
- `docs/results.md`: one table, one row per phase.

### 1. Processing element

- Integer datapath: int8 × int8 products into a wide accumulator (24 or 32 bits).
  Round and saturate once, at the array's output (in the VPU), not in every PE.
  Keep a Q8.8 build option for training if it is still needed.
- An optional pipeline register between the multiply and the add.
- No zeroing when not valid: registers hold their value (fewer toggles). Gate the
  operands (operand isolation) instead of clearing results.
- Synchronous reset on control registers only (valid, switch); none on the datapath.

### 2. Array

- Parameterised N × N (2 to 16), weight-stationary as upstream, with the double-buffered
  weights kept (load the next tile while the current one computes).
- Tang Nano 20K: about 8 × 8 with 9 × 9 multiplier halves (64 MACs, 96 available),
  accumulators in LUT/ALU.

### 3. Memory

- The unified buffer as banked single-port memories: one bank per array row, shared
  address generators instead of seven pointer sets.
- sky130: OpenRAM or DFFRAM macros; Tang Nano 20K: block RAM.
- Double buffering so loading and computing overlap.

### 4. Clock gating

- sky130: integrated clock gates (`sky130_fd_sc_hd__dlclkp`) per PE row and column and
  per VPU stage, in place of resetting idle PEs.
- Tang Nano 20K: clock enables (or DQCE) for the same groups.

### 5. Control

- An instruction memory and a small sequencer with loop instructions for tiling, so
  the host no longer streams a 130-bit word per cycle.

### 6. ASIC closure

- OpenLane configuration for speed: a 10 ns target first, then tightened; DELAY synthesis
  strategy.
- Power signed off with switching activity from the MNIST run.

## Status

The int8 inference core is in `rtl/` (every change from the baseline: `docs/CHANGES.md`).
Written and checked against the clock-level model (`model/ttf_model.py`), not yet simulated
as HDL or synthesized:

| Phase | State |
|---|---|
| 0. Baseline numbers | scripts for the new core ready (`fast.mk`: sim, tn20k, asic); to run locally, for both designs |
| 1. Processing element | done: int8, exact partial sums, PIPE option, enables, control-only reset |
| 2. Array | done: N = 4, 8, 16; diagonal weight load, no stall between tiles |
| 3. Memory | split into per-lane RAMs (inferred); SRAM macros for the ASIC still to do |
| 4. Clock gating | clock enables in place; gating cells still to do |
| 5. Control | done: descriptor RAM and sequencer |
| 6. ASIC closure | `asic/config.json` (10 ns), not run |

## Decisions

- **Number format:** int8, inference only (decided).
- **Array size on the ASIC:** open; set by the area budget (a Tiny Tapeout or Caravel slot,
  or a free size).
