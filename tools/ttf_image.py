#!/usr/bin/env python3
"""Writes a test image for sim/tb_ttf_core.v: the host-bus writes that load a network and
its inputs, and the words the outputs must read back.

    python3 tools/ttf_image.py [--model model/mnist_int8.json | --random SEED] [--m 8]
                               [--n 8] [--out build/sim]

Files: writes.memh (one "aaaadddddddd" per line, then ffff00000000), expect.memh (the same
format: the output words), config.txt (the parameters the testbench must be built with).
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import ttf_model as tm  # noqa: E402


def mnist(path, m, n):
    d = json.load(open(path))
    layers = [tm.Layer(l["w"], l["bias"], l["s0"], l["mult"], l["s1"], l["relu"]) for l in d["layers"]]
    s = d["samples"][:m]
    return layers, [x["x"] for x in s], [x["logits"] for x in s]


def rand_net(seed, m):
    rng = random.Random(seed)
    dims = [rng.randint(5, 70) for _ in range(rng.randint(2, 4))]
    layers = [tm.random_layer(rng, dims[i], dims[i + 1], i < len(dims) - 2)
              for i in range(len(dims) - 1)]
    x = [[rng.randint(-128, 127) for _ in range(dims[0])] for _ in range(m)]
    return layers, x, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.path.join(HERE, "..", "model", "mnist_int8.json"))
    ap.add_argument("--random", type=int, default=None, metavar="SEED")
    ap.add_argument("--m", type=int, default=8, help="batch (rows)")
    ap.add_argument("--n", type=int, default=8, help="array size of the build")
    ap.add_argument("--out", default="build/sim")
    a = ap.parse_args()
    if a.random is not None:
        layers, x, _ = rand_net(a.random, a.m)
    else:
        layers, x, want = mnist(a.model, a.m, a.n)
    img = tm.Image(layers, a.m, a.n)
    img.set_input(x)
    ref = tm.net_ref(layers, x)
    if a.random is None:
        assert ref == want, "the model file's logits differ from the reference arithmetic"
    exp_rows = {}
    for mi, row in enumerate(ref):
        for nb in range(img.out_blocks):
            lanes = row[nb * a.n:(nb + 1) * a.n]
            exp_rows[img.act_out + mi * img.out_blocks + nb] = lanes + [0] * (a.n - len(lanes))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "writes.memh"), "w") as f:
        for ad, d in tm.bus_writes(img):
            f.write("%04x%08x\n" % (ad, d))
        f.write("ffff00000000\n")
    q = a.n // 4
    with open(os.path.join(a.out, "expect.memh"), "w") as f:
        for ad in img.output_addrs():
            for i in range(q):
                f.write("%04x%08x\n" % (tm.A_ACT + ad * q + i, tm.pack4(exp_rows[ad][4 * i:4 * i + 4])))
        f.write("ffff00000000\n")
    macs = sum(a.m * ly.k * ly.n for ly in layers)
    with open(os.path.join(a.out, "config.txt"), "w") as f:
        f.write("N=%d W_ROWS=%d B_ROWS=%d M=%d MACS=%d\n" % (a.n, img.w_rows, img.b_rows, a.m, macs))
    print("%s: %d layers, batch %d, %d weight rows, %d bias rows, %d MACs"
          % (a.out, len(layers), a.m, img.w_rows, img.b_rows, macs))


if __name__ == "__main__":
    main()
