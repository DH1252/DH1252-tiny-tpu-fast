# What changed, explained from the start

This is for readers who are new to machine learning, to chip design, or to both. It explains what the original tiny-tpu does and what this project changed in it to make it faster and use less energy. Every section points to the matching entry (C1, C2, …) in [CHANGES.md](CHANGES.md), the technical version with file names and line numbers.

Nothing has been measured on real hardware yet. The numbers below come from a program that copies the new circuit clock tick by clock tick (`model/ttf_model.py`, see section 2.11).

---

## Part 1: the background you need

### 1.1 The job: recognizing handwritten digits

The demo reads a small picture of a handwritten digit, 0 to 9, and says which digit it is. The pictures come from MNIST, a well-known set of 70,000 such images.

- Each picture is 28 × 28 = 784 pixels.
- Each pixel is reduced to black (1) or white (0).

The recognizer is a small **neural network** with two **layers**:

```
784 pixels  →  layer 1  →  64 numbers  →  layer 2  →  10 scores, one per digit
```

The digit with the highest score is the answer.

### 1.2 What a layer computes

Every output of a layer is computed the same way:

1. Take every input.
2. Multiply each one by its own **weight**, a number learned during training.
3. Add up all the products.
4. Add one more learned number, the **bias**.
5. For layer 1, replace a negative result by 0. This step is called **ReLU**.

For example, an output with three inputs, weights 0.5, −1 and 2, and a bias of 0.25, given the inputs 1, 0, 1, is:

```
1 × 0.5  +  0 × (−1)  +  1 × 2  +  0.25  =  2.75
```

The basic step, "multiply two numbers and add the product to a running total", is called a **multiply-accumulate**, or **MAC**. One picture through this network takes:

```
784 × 64 + 64 × 10 = 50,816 MACs
```

Making the chip fast and efficient mostly means doing MACs quickly and cheaply.

### 1.3 Training versus inference

- **Training** is how the weights are found. You show the network many labelled pictures and nudge every weight to reduce its mistakes. This needs extra arithmetic: errors, and how much each weight is to blame.
- **Inference** is using the finished network. You put a picture in and get an answer out.

The original tiny-tpu can do both. This project does **inference only** (section 2.1). The training is done once, on a PC.

### 1.4 Chips: clock, registers and logic

A digital chip is made of two kinds of parts:

- **Registers.** Small boxes, each holding a number.
- **Logic.** Circuits between the registers that compute new values: adders, multipliers, comparisons.

A **clock** ticks millions of times a second, like a metronome. On every tick, every register takes in the new value that the logic in front of it has computed. Between ticks, the logic has to finish. The slowest chain of logic between two registers, the **critical path**, sets how fast the clock may tick.

> Clock speed is measured in MHz, millions of ticks per second. 100 MHz means a tick every 10 nanoseconds.

### 1.5 Memory

A chip can hold numbers in two ways:

- **In registers.** Fast, and readable all at once, but each bit takes a lot of space and power.
- **In RAM (memory blocks).** Many numbers packed densely. You read or write one row at a time by giving its **address** (its row number).

RAM is much cheaper per bit. In a custom chip, a register bit costs roughly ten times the area of a RAM bit.

### 1.6 Where the energy goes

A chip uses energy in two ways:

- **Switching (dynamic) energy.** Every time a wire changes from 0 to 1 or back, a tiny bit of electric charge is spent. The clock itself switches on every tick, and so does every register it reaches.
- **Leakage (static) energy.** A small constant drain for as long as the chip is powered.

So there are three ways to save energy:

1. Flip fewer bits for each useful calculation.
2. Don't let parts tick when they have nothing to do.
3. Finish sooner, so the constant drain has less time to add up.

### 1.7 The systolic array: a grid of small calculators

A TPU (Tensor Processing Unit, Google's name for this kind of chip) does many MACs at once with a grid of small calculators called **processing elements (PEs)**. Each PE holds one weight.

- **Inputs move to the right**, one PE per tick.
- **Running sums move down**, one PE per tick. Each PE adds "input × its weight" to the sum passing through it.
- **Finished sums come out of the bottom.**

It works like a bucket brigade: every PE does a small part and passes the work on, so all of them are busy at the same time. This arrangement is a **systolic array**, after the rhythmic pumping of a heart.

Because each PE keeps its weight while inputs flow past, it is called **weight-stationary**. The array only holds a small square of weights at a time, a **tile**. Big layers are cut into tiles, and the partial results are added up.

### 1.8 FPGA and ASIC

- **FPGA.** A chip that can be rewired by loading a configuration file. The first target here is the Tang Nano 20K, a small and cheap FPGA board. Good for testing.
- **ASIC.** A chip made specially for one design. Faster and far more efficient, but it costs money and months to make. The final goal.

---

## Part 2: what changed, and why

### 2.1 Only what recognition needs (C1)

The original chip contains circuits for training: error calculation, a second activation step, and weight updates. This project only recognizes, so all of that is gone. Circuits that don't exist take no space and use no energy.

### 2.2 Smaller numbers (C2, C3)

**Before:** every number was 16 bits in a format called **Q8.8**.

- 8 bits hold the whole part and 8 bits the fraction, so 2.75 is stored as 2.75 × 256 = 704.
- After *every* multiply and add, the result was rounded and limited ("saturated") to stay in range.

**Now:** weights and activations are **8-bit whole numbers** (int8, −128 to 127). This is called **quantization**.

- A real-valued weight like 0.0123 is stored as a whole number together with one **scale** per layer, for example "every weight means its value × 0.0004". The 0.0123 becomes 31.
- Sums inside the array are kept exactly, with no rounding at all.
- Only at the end of each output is the total converted back to an 8-bit number using the scales (the **requantization** step).

Why this helps:

- A multiplier's size, and the energy it uses, grows roughly with the *square* of the number width. An 8 × 8 multiplier is about a quarter of a 16 × 16 one.
- Half as many bits move through the array and the memories.
- The rounding circuits disappear from inside every PE.

Does the network get worse? Hardly. On the 10,000 MNIST test pictures:

| | Accuracy |
|---|---|
| Original 16-bit network | 97.35 % |
| New 8-bit network | 97.27 % |

The two agree on the answer for 99.70 % of the pictures. The conversion is done by `tools/ttf_quant_mnist.py`.

### 2.3 More calculators (C8)

**Before:** a 2 × 2 grid, 4 MACs per tick at most.

**Now:** the grid size is a setting: 4 × 4, 8 × 8 (64 MACs per tick, the Tang Nano 20K build) or 16 × 16 (256).

More calculators only help if they are kept fed. That depends on the **batch**: how many pictures you process together.

- **One picture at a time.** Every weight is used once and thrown away. The grid spends most of its time waiting for weights. An 8 × 8 grid manages about 8 MACs per tick: 6,470 ticks per picture.
- **Eight pictures at a time.** Each weight, once loaded, is used for all eight. The 8 × 8 grid manages 62.8 of its 64 possible MACs per tick: 809 ticks per picture.

If the board reaches 100 MHz, 809 ticks per picture means about 120,000 pictures per second for the core. Getting pictures in and out over the USB serial link is slower than that.

### 2.4 Loading the next weights while still working (C5)

**Before:**

- Each PE had two weight registers: the one in use, and a spare for the next tile.
- New weights were pushed down through every PE in the column, so every weight register in the column changed on every loading tick.
- A "swap" signal then switched spare and active.

**Now:** each PE has one weight register, and it is written exactly once per tile, at exactly the right tick.

- A one-bit "load now" marker travels down each column, one PE per tick.
- The weight for that row waits on a wire running down the column.
- The marker reaches each PE just after the last input of the old tile has passed it, and just before the first input of the new tile arrives.

Think of an orchestra changing songs. Instead of every musician swapping all their sheet music at once, an assistant walks along the rows and replaces each page right after that musician plays the last note of the old song. The new song can start immediately, and nobody waits.

Result: half the weight registers, no weights bouncing through other PEs, and no pause between tiles.

### 2.5 Not ticking when there's nothing to do (C6, C7, C19)

**Before:**

- A PE without new data actively cleared its outputs to zero on every tick, so it switched bits even while idle.
- Every register had a reset wire.

**Now:**

- **Hold, don't clear.** Registers only take a new value when there is real data. Otherwise they keep what they have, and nothing flips.
- **Reset only what needs it.** Only the few control flags that must start in a known state get a reset. Data registers don't need one: they are always written before they are read. This makes each register smaller and removes a wire to every data bit.
- **Stop the clock entirely (C19).** On the Tang Nano 20K, the core's clock is switched off at its source whenever the core has nothing to do. The FPGA's clock buffer does this. Only the small serial-port bridge keeps ticking. LED 4 lights while the core's clock runs. Between pictures, the core spends almost no switching energy.

### 2.6 Splitting the big shelf into drawers (C9, C10)

**Before:** one shared store, the "unified buffer", held everything (inputs, weights, biases, results) in 128 registers. Seven separate sets of counters kept track of what to read where.

**Now:** separate RAM blocks, one per job:

- inputs and results (activations);
- weights;
- the running totals of each column;
- the biases;
- the list of layers to run.

The activation and weight memories are further split into one RAM per **lane** (per row or column of the grid).

That split solves a timing puzzle. In a systolic array, row 2 must receive its input one tick after row 1, row 3 one tick after row 2, and so on. This is called **skew**.

- **Before**, the skew was made by passing the data itself through extra registers.
- **Now**, each lane's own RAM is simply asked one tick later than the previous lane's. Only the *address*, which changes once per picture row, is delayed. The data goes straight from the RAM into the grid.

### 2.7 Finishing each result at the bottom of its column (C11, C12)

Under each column sits a small **output lane**. It:

1. adds the bias;
2. adds up the partial results of all the tiles a layer was cut into;
3. converts the total back to an 8-bit number using the layer's scale;
4. applies ReLU;
5. writes the result straight into the memory the next layer reads its inputs from. Nothing has to be copied.

Before, the first two jobs were done outside the TPU, by extra logic in the demo.

### 2.8 A to-do list instead of step-by-step orders (C13, C14)

**Before:** the computer driving the chip had to supply a new 130-bit instruction on *every* tick.

**Now:** you write one short description per layer into the chip, eight numbers each: where its inputs are, how big it is, its scale, whether it uses ReLU. Then you press "start" once. A small **sequencer** walks through all the layers by itself and raises a "done" flag at the end.

The PC is free while the network runs, and its connection is no longer the bottleneck. A counter (CYCLES) reports how many ticks the run took.

### 2.9 Shorter steps for a faster beat (C4)

The clock can only tick as fast as the slowest step between two registers allows (section 1.4).

**Before:** a PE multiplied, rounded, limited, added and limited again in one tick.

**Now:** with the PIPE setting on (the default), the PE multiplies in one tick and adds in the next, with a register in between. Each step is shorter, so the clock can run faster.

This costs one extra tick of delay for the whole grid, not one per PE.

### 2.10 The board and the build settings (C15, C16, C18)

- **The board.** The Tang Nano 20K talks to a PC over USB. A small command set is enough to drive everything: write a number, write many numbers, read a number, ping.
- **The PC program.** `tools/ttf_host.py` loads the network, sends pictures, and checks every answer against the expected one, bit for bit.
- **The clock.** It comes from the FPGA's frequency multiplier (PLL), so you can ask for, say, 100 MHz.
- **The build settings.** The FPGA tool (Gowin EDA) has settings for how hard it should work on timing. These are now set for speed:
  - aim slightly above the target clock;
  - put multiplications into the FPGA's dedicated multiplier blocks, which are faster and much more frugal than building multipliers from general logic;
  - place and route the circuit with timing first.

  Settings that would cost energy without making it faster were left out.
- **The ASIC.** A configuration for the free OpenLane chip-design flow asks for a 10 ns clock (100 MHz). The original asked for 50 ns.

Speed and energy are not opposites here. Energy per picture is power × time. A design that switches little (section 2.5) and finishes quickly uses less energy per picture.

### 2.11 How we know it works, and what isn't checked yet (C17)

- **A software copy of the chip.** `model/ttf_model.py` copies every register and memory of the new circuit, tick by tick. It is run on random networks of several shapes and on the real MNIST network, in every configuration, and its outputs are compared with the arithmetic done the plain way. They match exactly.
- **The check catches mistakes.** Deliberately shifting a timing signal by one tick makes the comparison fail, so it would notice a real timing error.
- **Still to come.** A test of the actual circuit description in a hardware simulator (`sim/tb_ttf_core.v`), and the first real build on the board. Those runs are the next step, and they can still find mistakes that the software copy shares with the circuit.

---

## Part 3: summary

| | Original tiny-tpu | This project |
|---|---|---|
| Purpose | training and inference | inference only |
| Number format | 16-bit Q8.8, rounded at every step | 8-bit whole numbers, exact sums, rounded once per output |
| Calculators (PEs) | 4 (2 × 2) | 64 (8 × 8) on the board; 16 or 256 possible |
| MACs per tick, 8 pictures at once | at most 4 | 62.8 |
| Weight loading | through every PE, then swap | written in place, just in time, no pause |
| Idle behaviour | registers cleared every tick | registers hold; the core's clock stops |
| Memory | one shared 128-register store | separate RAM blocks, one per lane |
| Control | a 130-bit instruction every tick from the PC | one description per layer, one start |
| MNIST accuracy | 97.35 % | 97.27 % |

## Glossary

| Word | Meaning |
|---|---|
| Activation | a number flowing between layers (an output of one layer, an input of the next) |
| ASIC | a chip made for one design |
| Batch | the number of pictures processed together |
| Bias | a learned number added to each output of a layer |
| Clock | the signal that ticks and makes all registers update together |
| Critical path | the slowest chain of logic between two registers; it limits the clock speed |
| FPGA | a chip that can be rewired by loading a file |
| Inference | running a trained network on new input |
| int8 | an 8-bit whole number, −128 to 127 |
| Lane | one row (input) or column (output) of the grid, with its own memory |
| Layer | one step of the network: weighted sums of its inputs, plus bias, then ReLU |
| MAC | multiply-accumulate: multiply two numbers and add the product to a total |
| MHz | millions of clock ticks per second |
| PE | processing element: one small calculator in the grid |
| PLL | the circuit that makes a faster clock from the board's 27 MHz crystal |
| Q8.8 | a 16-bit number with 8 bits before and 8 bits after the binary point |
| Quantization | storing numbers with fewer bits plus a scale |
| RAM | memory read and written one row at a time, by address |
| Register | a small storage box that takes a new value on each clock tick |
| ReLU | "replace a negative number by 0" |
| Requantization | converting a wide total back to an 8-bit number using the scales |
| Skew | starting each row of the grid one tick after the previous row |
| Systolic array | a grid of PEs that pass data to their neighbours on each tick |
| Tile | the square of weights the grid holds at one time |
| Training | finding the weights from labelled examples |
| Weight | a learned number that multiplies an input |
