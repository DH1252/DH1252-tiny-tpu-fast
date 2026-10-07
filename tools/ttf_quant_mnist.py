#!/usr/bin/env python3
"""Converts the upstream MNIST model (Q8.8, mnist_demo/data/model/reference) to the int8
inference core's format and measures what the conversion costs in accuracy.

    python3 tools/ttf_quant_mnist.py --mnist DIR [--out model/mnist_int8.json]

DIR holds the four MNIST idx files (t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz,
train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz). Inputs are binarized as upstream
evaluates them: pixel / 255 > 50 / 255 (summary.json eval_threshold).

Quantization (post-training, per layer, symmetric):
  inputs   x in {0, 1}, scale 1
  weights  w_q = round(w / s_w), s_w = max|w| / 127
  biases   b_q = round(b / (s_x * s_w)), int32
  outputs  scale s_y from the largest activation over the calibration images (training
           set), requantization parameters (s0, mult, s1) so that
           y = sat8(rnd(sat18(rnd(acc, s0)) * mult, s1)) ~ acc * s_x * s_w / s_y
The last layer gives int8 logits; the class is their argmax (first maximum).
Needs numpy.
"""

import argparse, gzip, json, os, struct, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "mnist_demo", "data", "model", "reference")
THRESHOLD_RAW = 50


def memh(name):
    vals = [int(l.strip(), 16) for l in open(os.path.join(REF, name)) if l.strip()]
    return np.array([v - 0x10000 if v & 0x8000 else v for v in vals], dtype=np.int64)


def untile(flat, k, n, tile=2):
    """upstream's flatten_weights_for_tiles (mnist_tools.py) inverted: K x N"""
    w = np.zeros((k, n), dtype=np.int64)
    i = 0
    for t0 in range(0, n, tile):
        t1 = min(t0 + tile, n)
        for kk in range(k):
            for o in range(t0, t1):
                w[kk, o] = flat[i]
                i += 1
    return w


def load_reference():
    w1 = untile(memh("w1_tiled_q8_8.memh"), 784, 64) / 256.0
    w2 = untile(memh("w2_tiled_q8_8.memh"), 64, 10) / 256.0
    b1 = memh("b1_q8_8.memh") / 256.0
    b2 = memh("b2_q8_8.memh") / 256.0
    return w1, b1, w2, b2


def idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        dims = struct.unpack(">" + "I" * (magic & 0xFF), f.read(4 * (magic & 0xFF)))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist(d, split):
    pre = "t10k" if split == "test" else "train"
    x = idx(os.path.join(d, pre + "-images-idx3-ubyte.gz")).reshape(-1, 784)
    y = idx(os.path.join(d, pre + "-labels-idx1-ubyte.gz"))
    return (x > THRESHOLD_RAW).astype(np.int64), y.astype(np.int64)


def q88(v):
    return np.clip(np.round(v * 256.0), -32768, 32767) / 256.0


def ref_forward(x, w1, b1, w2, b2):
    """the Q8.8 network as trained (fake quantization after each linear layer)"""
    h = np.maximum(q88(x @ w1 + b1), 0.0)
    return h, q88(h @ w2 + b2)


# ---------------------------------------------------------------- the core's arithmetic

def rnd(x, s):
    return (x + ((1 << s) >> 1)) >> s


def requant(acc, s0, mult, s1, relu):
    ys = np.clip(rnd(acc, s0), -(1 << 17), (1 << 17) - 1)
    z = rnd(ys * mult, s1)
    if relu:
        z = np.maximum(z, 0)
    return np.clip(z, -128, 127)


def int_forward(x, layers):
    a = x
    accs = []
    for ly in layers:
        acc = a @ np.array(ly["w"], dtype=np.int64) + np.array(ly["bias"], dtype=np.int64)
        accs.append(acc)
        a = requant(acc, ly["s0"], ly["mult"], ly["s1"], ly["relu"])
    return a, accs


def requant_params(m_real, acc_max):
    """(s0, mult, s1) with mult / 2^(s0 + s1) ~ m_real and |acc| >> s0 within 18 bits"""
    s0 = 0
    while (acc_max >> s0) > (1 << 17) - 1:
        s0 += 1
    assert s0 < 16, "accumulator too large for the 18-bit multiplier input"
    best = None
    for s1 in range(32):
        mult = int(round(m_real * (1 << (s0 + s1))))
        if mult >= (1 << 17):
            break
        best = (s0, mult, s1)
    assert best and best[1] > 0, "requantization multiplier out of range"
    return best


def quantize(w1, b1, w2, b2, xcal):
    layers = []
    s_x = 1.0
    h_cal, l_cal = ref_forward(xcal, w1, b1, w2, b2)
    for w, b, y_cal, relu in ((w1, b1, h_cal, 1), (w2, b2, l_cal, 0)):
        s_w = np.abs(w).max() / 127.0
        wq = np.clip(np.round(w / s_w), -128, 127).astype(np.int64)
        bq = np.round(b / (s_x * s_w)).astype(np.int64)
        s_y = np.abs(y_cal).max() / 127.0
        layers.append(dict(w=wq.tolist(), bias=bq.tolist(), relu=relu,
                           s_in=s_x, s_w=float(s_w), s_out=float(s_y)))
        s_x = s_y
    # accumulator ranges on the calibration set fix s0; then mult, s1
    a = xcal
    for ly in layers:
        acc = a @ np.array(ly["w"], dtype=np.int64) + np.array(ly["bias"], dtype=np.int64)
        m_real = ly["s_in"] * ly["s_w"] / ly["s_out"]
        ly["s0"], ly["mult"], ly["s1"] = requant_params(m_real, int(np.abs(acc).max()) * 2)
        a = requant(acc, ly["s0"], ly["mult"], ly["s1"], ly["relu"])
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mnist", required=True, help="directory with the MNIST idx .gz files")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "model", "mnist_int8.json"))
    ap.add_argument("--calib", type=int, default=10000, help="training images for calibration")
    ap.add_argument("--samples", type=int, default=64, help="test images kept in the output file")
    a = ap.parse_args()

    w1, b1, w2, b2 = load_reference()
    xtr, _ = load_mnist(a.mnist, "train")
    xte, yte = load_mnist(a.mnist, "test")
    layers = quantize(w1, b1, w2, b2, xtr[:a.calib])

    _, lref = ref_forward(xte, w1, b1, w2, b2)
    pref = lref.argmax(1)
    logits, accs = int_forward(xte, layers)
    pint = logits.argmax(1)
    acc_ref = float((pref == yte).mean())
    acc_int = float((pint == yte).mean())
    agree = float((pref == pint).mean())
    acc_bits = [int(np.abs(x).max()).bit_length() + 1 for x in accs]
    print("test images        %d" % len(yte))
    print("Q8.8 reference     %.4f" % acc_ref)
    print("int8 core          %.4f" % acc_int)
    print("same class         %.4f" % agree)
    print("accumulator bits   %s (largest |acc| on the test set, signed)" % acc_bits)
    for i, ly in enumerate(layers):
        print("layer %d: %d x %d  s0 %d  mult %d  s1 %d  relu %d" %
              (i, len(ly["w"]), len(ly["w"][0]), ly["s0"], ly["mult"], ly["s1"], ly["relu"]))

    out = dict(
        source="mnist_demo/data/model/reference (Q8.8), post-training int8 quantization",
        input="784 pixels, 1 if pixel > %d else 0 (raw 0..255)" % THRESHOLD_RAW,
        test_images=len(yte), accuracy_q88=acc_ref, accuracy_int8=acc_int, same_class=agree,
        accumulator_bits=acc_bits,
        layers=[dict(w=ly["w"], bias=ly["bias"], s0=ly["s0"], mult=ly["mult"], s1=ly["s1"],
                     relu=ly["relu"], s_in=ly["s_in"], s_w=ly["s_w"], s_out=ly["s_out"])
                for ly in layers],
        samples=[dict(x=xte[i].tolist(), label=int(yte[i]), logits=logits[i].tolist(),
                      cls=int(pint[i])) for i in range(a.samples)])
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
