#!/usr/bin/env python3
"""Runs the int8 MNIST network on the Tang Nano 20K (boards/tn20k) and checks every result
against the reference arithmetic.

    python3 tools/ttf_host.py --port /dev/ttyUSB1 [--baud 115200] [--mhz 27] [--m 8]
                              [--mnist DIR] [--images 1000]

Without --mnist it runs the 64 test images stored in model/mnist_int8.json; with it, the
first --images images of the MNIST test set (and reports the accuracy). For each batch of
--m images: write the inputs, run, read CYCLES and the logits. Prints clocks per batch,
MAC per clock and, with --mhz (the build's clock), images per second of the core alone.
Needs pyserial.
"""
import argparse, gzip, json, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import ttf_model as tm  # noqa: E402


class Bus:
    """the board's byte protocol (boards/tn20k/ttf_tn20k_top.v)"""

    MAX_READS = 16      # reply FIFO: 64 bytes

    def __init__(self, port, baud):
        import serial
        self.s = serial.Serial(port, baud, timeout=2)
        time.sleep(0.1)
        self.s.reset_input_buffer()

    def _rx(self, n):
        b = self.s.read(n)
        if len(b) != n:
            raise IOError("no answer from the board (%d of %d bytes): port, baud rate?" % (len(b), n))
        return b

    def ping(self):
        self.s.write(b"P")
        return self._rx(1) == b"K"

    def write(self, a, d):
        self.s.write(b"W" + struct.pack("<HI", a, d & 0xFFFFFFFF))
        if self._rx(1) != b"K":
            raise IOError("write not acknowledged")

    def burst(self, a, words):
        """consecutive words from address a, up to 256 per command, one ack each"""
        pend = 0
        for i in range(0, len(words), 256):
            chunk = words[i:i + 256]
            self.s.write(b"B" + struct.pack("<HB", a + i, len(chunk) & 0xFF)
                         + b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in chunk))
            pend += 1
            if pend == 4:                 # keep the BL616's buffer from overflowing
                if self._rx(pend) != b"K" * pend:
                    raise IOError("burst not acknowledged")
                pend = 0
        if pend and self._rx(pend) != b"K" * pend:
            raise IOError("burst not acknowledged")

    def writes(self, pairs):
        """(address, data) pairs: runs of consecutive addresses go as bursts"""
        run_a, run = None, []
        for a, d in pairs:
            if run and a == run_a + len(run):
                run.append(d)
                continue
            if run:
                self.burst(run_a, run)
            run_a, run = a, [d]
        if run:
            self.burst(run_a, run)

    def read(self, a):
        return self.reads([a])[0]

    def reads(self, addrs):
        out = []
        for i in range(0, len(addrs), self.MAX_READS):
            chunk = addrs[i:i + self.MAX_READS]
            self.s.write(b"".join(b"R" + struct.pack("<H", a) for a in chunk))
            raw = self._rx(4 * len(chunk))
            out += list(struct.unpack("<%dI" % len(chunk), raw))
        return out


def load_test(d):
    def idx(name):
        with gzip.open(os.path.join(d, name), "rb") as f:
            magic, = struct.unpack(">I", f.read(4))
            dims = struct.unpack(">" + "I" * (magic & 0xFF), f.read(4 * (magic & 0xFF)))
            return f.read(), dims
    xb, dims = idx("t10k-images-idx3-ubyte.gz")
    yb, _ = idx("t10k-labels-idx1-ubyte.gz")
    x = [[1 if xb[i * 784 + j] > 50 else 0 for j in range(784)] for i in range(dims[0])]
    return x, list(yb)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mhz", type=float, default=27.0, help="the build's core clock (CLK_MHZ)")
    ap.add_argument("--m", type=int, default=8, help="images per run (the batch)")
    ap.add_argument("--model", default=os.path.join(HERE, "..", "model", "mnist_int8.json"))
    ap.add_argument("--mnist", help="directory with the MNIST test idx .gz files")
    ap.add_argument("--images", type=int, default=1000)
    a = ap.parse_args()

    bus = Bus(a.port, a.baud)
    if not bus.ping():
        sys.exit("no answer to ping")
    ident, params, sizes = bus.reads([tm.A_ID, tm.A_PARAMS, tm.A_SIZES])
    if ident != 0x54544631:
        sys.exit("ID %08x: not the tiny-tpu-fast core" % ident)
    n, g, acc_w, pipe = params & 0xFF, (params >> 8) & 0xFF, (params >> 16) & 0xFF, (params >> 24) & 1
    aw_m = (sizes >> 10) & 31
    print("core: N %d, G %d, ACC_W %d, PIPE %d, batch up to %d" % (n, g, acc_w, pipe, 1 << aw_m))
    if a.m > (1 << aw_m):
        sys.exit("--m %d is larger than the build's batch (%d)" % (a.m, 1 << aw_m))

    d = json.load(open(a.model))
    layers = [tm.Layer(l["w"], l["bias"], l["s0"], l["mult"], l["s1"], l["relu"]) for l in d["layers"]]
    if a.mnist:
        xs, ys = load_test(a.mnist)
        xs, ys = xs[:a.images], ys[:a.images]
    else:
        xs, ys = [s["x"] for s in d["samples"]], [s["label"] for s in d["samples"]]

    img = tm.Image(layers, a.m, n)
    t0 = time.time()
    bus.writes(tm.bus_writes(img))
    print("weights, biases, descriptors loaded: %.1f s" % (time.time() - t0))

    macs_img = sum(ly.k * ly.n for ly in layers)
    tot = dict(img=0, ok=0, same=0, cyc=0)
    for b0 in range(0, len(xs), a.m):
        batch = xs[b0:b0 + a.m]
        while len(batch) < a.m:
            batch = batch + [batch[-1]]
        img.act = {}
        img.set_input(batch)
        bus.writes(tm.bus_writes(img, inputs_only=True))
        bus.write(tm.A_CTRL, 1)
        while not (bus.read(tm.A_STATUS) & 2):
            pass
        cyc = bus.read(tm.A_CYCLES)
        got = tm.unpack_output(img, bus.reads(tm.bus_output_words(img)))
        want = tm.net_ref(layers, batch)
        for i in range(min(a.m, len(xs) - b0)):
            tot["img"] += 1
            tot["same"] += got[i] == want[i]
            cls = max(range(len(got[i])), key=lambda j: (got[i][j], -j))
            tot["ok"] += cls == ys[b0 + i]
        tot["cyc"] += cyc
        if b0 == 0:
            print("batch of %d: %d clocks, %.1f MAC/clock of %d, %.0f images/s at %g MHz"
                  % (a.m, cyc, macs_img * a.m / cyc, n * n, a.m * a.mhz * 1e6 / cyc, a.mhz))
    runs = -(-tot["img"] // a.m)
    print("%d images: %d match the reference bit for bit, accuracy %.4f, %.1f clocks per image"
          % (tot["img"], tot["same"], tot["ok"] / tot["img"], tot["cyc"] / (runs * a.m)))
    sys.exit(0 if tot["same"] == tot["img"] else 1)


if __name__ == "__main__":
    main()
