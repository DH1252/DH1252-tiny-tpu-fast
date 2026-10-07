"""tiny-tpu-fast: reference model of the int8 inference core (rtl/ttf_*.v).

Three parts:
  - layer_ref(): what a layer computes, bit-exact (the numbers the hardware must produce);
  - Image: the memory layout of a network (weight rows, bias rows, descriptors, inputs),
    as the host writes it over the bus (tools/ttf_host.py) and the testbench loads it;
  - CoreSim: a clock-by-clock mirror of the RTL's registers and RAMs, to check the timing
    scheme (diagonal weight loading, lane skews, output lanes) against layer_ref() without
    an HDL simulator.

Run it to self-check: python3 model/ttf_model.py
"""

import random

# ------------------------------------------------------------------------------- numerics


def wrap(x, bits):
    """two's complement wrap to a signed bits-wide value"""
    m = 1 << bits
    x &= m - 1
    return x - m if x >> (bits - 1) else x


def rnd(x, s):
    """round half up, then arithmetic shift: (x + (2^s >> 1)) >> s"""
    return (x + ((1 << s) >> 1)) >> s


def sat(x, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo if x < lo else hi if x > hi else x


def requant(acc, s0, mult, s1, relu):
    ys = sat(rnd(acc, s0), 18)
    z = rnd(ys * mult, s1)
    if relu and z < 0:
        return 0
    return sat(z, 8)


class Layer:
    """out[m][n] = requant(sum_k x[m][k] * w[k][n] + bias[n]); w is K x Nout, int8"""

    def __init__(self, w, bias, s0, mult, s1, relu):
        self.w, self.bias = w, bias
        self.k, self.n = len(w), len(w[0])
        self.s0, self.mult, self.s1, self.relu = s0, mult, s1, relu
        assert 0 <= s0 < 16 and 0 <= mult < (1 << 17) and 0 <= s1 < 32


def layer_ref(layer, x, acc_w=32):
    out = []
    for row in x:
        o = []
        for n in range(layer.n):
            acc = layer.bias[n] + sum(row[k] * layer.w[k][n] for k in range(layer.k))
            o.append(requant(wrap(acc, acc_w), layer.s0, layer.mult, layer.s1, layer.relu))
        out.append(o)
    return out


def net_ref(layers, x, acc_w=32):
    for ly in layers:
        x = layer_ref(ly, x, acc_w)
    return x


# ------------------------------------------------------------------------------- layout


def cdiv(a, b):
    return -(-a // b)


class Image:
    """Memory contents for a network on an N x N core.

    act:  {address: [N lanes of int8]}   inputs at act_in, outputs of the last layer at act_out
    wgt:  {row: [N lanes of int8]}
    bias: {row: [N lanes of int32]}
    desc: [32-bit words]                 8 per layer
    Layer i reads ACT from a_base[i], writes o_base[i]; consecutive layers ping-pong
    between two regions so a layer never overwrites its own input.
    """

    def __init__(self, layers, m, n, act_words=2048):
        self.n, self.m = n, m
        self.act, self.wgt, self.bias, self.desc = {}, {}, {}, []
        self.layers = layers
        regions = [0, act_words // 2]
        w_row = b_row = 0
        self.blocks = []
        for i, ly in enumerate(layers):
            kb_n, nb_n = cdiv(ly.k, n), cdiv(ly.n, n)
            a_base, o_base = regions[i % 2], regions[(i + 1) % 2]
            assert m * max(kb_n, nb_n) <= act_words // 2, "activations do not fit"
            for nb in range(nb_n):
                for kb in range(kb_n):
                    for r in range(n):
                        k = kb * n + r
                        self.wgt[w_row + (nb * kb_n + kb) * n + r] = [
                            ly.w[k][nb * n + c] if k < ly.k and nb * n + c < ly.n else 0
                            for c in range(n)]
                self.bias[b_row + nb] = [ly.bias[nb * n + c] if nb * n + c < ly.n else 0
                                         for c in range(n)]
            last = 1 if i == len(layers) - 1 else 0
            self.desc += [
                (a_base & 0xFFFF) | (o_base & 0xFFFF) << 16,
                w_row,
                kb_n | nb_n << 12,
                m | b_row << 12,
                ly.mult,
                ly.s0 | ly.s1 << 4 | ly.relu << 9 | last << 10,
                0, 0]
            self.blocks.append((a_base, kb_n, o_base, nb_n))
            w_row += nb_n * kb_n * n
            b_row += nb_n
        self.w_rows, self.b_rows = w_row, b_row
        self.act_in = self.blocks[0][0]
        self.act_out, self.out_blocks = self.blocks[-1][2], self.blocks[-1][3]

    def set_input(self, x):
        a_base, kb_n = self.blocks[0][0], self.blocks[0][1]
        k = self.layers[0].k
        for mi, row in enumerate(x):
            for kb in range(kb_n):
                self.act[a_base + mi * kb_n + kb] = [
                    row[kb * self.n + r] if kb * self.n + r < k else 0 for r in range(self.n)]

    def output_addrs(self):
        return [self.act_out + mi * self.out_blocks + nb
                for mi in range(self.m) for nb in range(self.out_blocks)]

    def read_output(self, act):
        """act: address -> lanes; the last layer's output as m rows of n values"""
        n_out = self.layers[-1].n
        out = []
        for mi in range(self.m):
            row = []
            for nb in range(self.out_blocks):
                row += act[self.act_out + mi * self.out_blocks + nb]
            out.append(row[:n_out])
        return out


# ------------------------------------------------------------------------------- RTL mirror


class Dly:
    """ttf_dly: tap(k) = input k clocks ago (k = 1 .. L)"""

    def __init__(self, length, init=0):
        self.t = [init] * length

    def tap(self, k):
        return self.t[k - 1]

    def step(self, x):
        self.t = [x] + self.t[:-1]


class Ram:
    """ttf_ram: registered read (holds when re = 0), read-before-write"""

    def __init__(self, depth, init=0):
        self.mem = [init] * depth
        self.q = init
        self._n = None

    def cycle(self, we, wa, wd, re, ra):
        q = self.mem[ra] if re else self.q
        if we:
            self.mem[wa] = wd
        self.q = q


class CoreSim:
    """Clock-level mirror of ttf_core (all RAMs and registers). Load an Image, run()."""

    def __init__(self, n=8, g=1, pipe=1, acc_w=32, aw_a=11, aw_w=13, aw_m=6, aw_b=8, aw_d=6):
        assert n in (4, 8, 16) and g in (1, 2, 4) and n % g == 0
        self.n, self.g, self.pipe, self.acc_w = n, g, pipe, acc_w
        self.aw_a, self.aw_w, self.aw_m, self.aw_b, self.aw_d = aw_a, aw_w, aw_m, aw_b, aw_d
        self.pw = 16 + (n.bit_length() - 1)
        self.out_d = n + 2 + pipe
        self.drain = self.out_d + n + 8
        self.act = [Ram(1 << aw_a) for _ in range(n)]
        self.wgt = [Ram(1 << aw_w, [0] * g) for _ in range(n // g)]
        self.acc = [Ram(1 << aw_m) for _ in range(n)]
        self.bia = [Ram(1 << aw_b) for _ in range(n)]
        self.dsc = Ram(1 << aw_d)
        self.cycles = 0

    def load(self, img):
        n, g = self.n, self.g
        assert img.n == n
        for a, lanes in img.act.items():
            for r in range(n):
                self.act[r].mem[a] = lanes[r]
        for row, lanes in img.wgt.items():
            for gi in range(n // g):
                self.wgt[gi].mem[row] = lanes[gi * g:(gi + 1) * g]
        for row, lanes in img.bias.items():
            for c in range(n):
                self.bia[c].mem[row] = lanes[c]
        for i, w in enumerate(img.desc):
            self.dsc.mem[i] = w

    def act_lanes(self, addr):
        return [wrap(self.act[r].mem[addr], 8) for r in range(self.n)]

    def run(self, max_cycles=10 ** 7):
        n, g, pipe, pw = self.n, self.g, self.pipe, self.pw
        ma, mw, mb = (1 << self.aw_a) - 1, (1 << self.aw_w) - 1, (1 << self.aw_b) - 1
        mm = (1 << self.aw_m) - 1
        ng = n // g
        # sequencer
        S = dict(st="LD", ld=0, dent=0, last=0, a_base=0, w_tile=0, kb_n=0, nb_n=0, m_n=0,
                 kb=0, nb=0, s=0, p_n=0, a_row=0, dr=0, busy=1, done=0, d_re=1, d_ra=0,
                 e_wv=0, e_wa=0, e_wtok=0, e_av=0, e_aa=0, e_fr=0, e_fk=0, e_lk=0, e_fl=0,
                 o_base=0, nb_o=0, b_base=0, s0=0, mult=0, s1=0, relu=0)
        ad, av, tk = Dly(n, (0, 0)), Dly(n + 1), Dly(n)
        fl = Dly(self.out_d + n - 1, (0, 0, 0, 0, 0))
        wd = Dly(max(n - g, 1), (0, 0))
        lane_d = {}                       # (group, l) -> Dly of l stages
        for gi in range(ng):
            for l in range(1, g):
                lane_d[(gi, l)] = Dly(l)
        # array
        W = [[0] * n for _ in range(n)]
        AO = [[0] * n for _ in range(n)]
        AV = [[0] * n for _ in range(n)]
        PS = [[0] * n for _ in range(n)]
        TK = [[0] * n for _ in range(n)]
        PP = [[0] * n for _ in range(n)]
        # output lanes
        P = [dict(m_cnt=0, nb_cnt=0, oa_reg=0, v1=0, fk1=0, lk1=0, m1=0, oa1=0,
                  acc_we=0, acc_wa=0, acc_wd=0, r0v=0, r0=0, oa_r0=0, r1v=0, ys=0, oa_r1=0,
                  r2v=0, p=0, oa_r2=0, o_we=0, o_wa=0, o_wd=0) for _ in range(n)]
        cyc = 0
        while True:
            cyc += 1
            if cyc > max_cycles:
                raise RuntimeError("no done")
            s = S
            # ---------------- combinational values of this clock
            w_bus = [0] * n
            for gi in range(ng):
                q = self.wgt[gi].q
                for l in range(g):
                    c = gi * g + l
                    w_bus[c] = q[0] if l == 0 else lane_d[(gi, l)].tap(l)
            # a lane's l-delay taps the RAM byte l of the group: handled in the step below
            tok = [tk.tap(c + 1) for c in range(n)]
            a_in = [wrap(self.act[r].q, 8) for r in range(n)]
            a_v = [av.tap(r + 2) for r in range(n)]
            ps_bot = [PS[n - 1][c] for c in range(n)]
            # ---------------- array next state
            nW = [row[:] for row in W]
            nAO = [row[:] for row in AO]
            nAV = [[0] * n for _ in range(n)]
            nPS = [row[:] for row in PS]
            nTK = [[0] * n for _ in range(n)]
            nPP = [row[:] for row in PP]
            for r in range(n):
                for c in range(n):
                    ai = a_in[r] if c == 0 else AO[r][c - 1]
                    vi = a_v[r] if c == 0 else AV[r][c - 1]
                    pi = 0 if r == 0 else PS[r - 1][c]
                    li = tok[c] if r == 0 else TK[r - 1][c]
                    if li:
                        nW[r][c] = wrap(w_bus[c], 8)
                    nAV[r][c] = vi
                    nTK[r][c] = li
                    if vi:
                        nAO[r][c] = ai
                    prod = ai * W[r][c]
                    if pipe:
                        if vi:
                            nPP[r][c] = prod
                        if AV[r][c]:
                            nPS[r][c] = wrap(pi + PP[r][c], pw)
                    elif vi:
                        nPS[r][c] = wrap(pi + prod, pw)
            # ---------------- output lanes
            act_wr = []
            for c in range(n):
                p = P[c]
                fv, ffr, ffk, flk, ffl = fl.tap(self.out_d + c - 1)
                new_nb = ffr and ffk
                m_cur = 0 if ffr else p["m_cnt"]
                nb_cur = ((0 if ffl else (p["nb_cnt"] + 1) & mb) if new_nb else p["nb_cnt"])
                oa_cur = ((s["o_base"] + nb_cur) & ma) if ffr else (p["oa_reg"] + s["nb_o"]) & ma
                base = self.bia[c].q if p["fk1"] else self.acc[c].q
                sm = wrap(base + ps_bot[c], self.acc_w)
                np_ = dict(p)
                if fv:
                    np_["m_cnt"] = (m_cur + 1) & mm
                    np_["oa_reg"] = oa_cur
                    if new_nb:
                        np_["nb_cnt"] = nb_cur
                np_["v1"] = fv
                if fv:
                    np_["fk1"], np_["lk1"], np_["m1"], np_["oa1"] = ffk, flk, m_cur, oa_cur
                np_["acc_we"] = p["v1"] and not p["lk1"]
                if p["v1"] and not p["lk1"]:
                    np_["acc_wa"], np_["acc_wd"] = p["m1"], sm
                np_["r0v"] = p["v1"] and p["lk1"]
                if p["v1"] and p["lk1"]:
                    np_["r0"], np_["oa_r0"] = sm, p["oa1"]
                np_["r1v"] = p["r0v"]
                if p["r0v"]:
                    np_["ys"] = sat(rnd(p["r0"], s["s0"]), 18)
                    np_["oa_r1"] = p["oa_r0"]
                np_["r2v"] = p["r1v"]
                if p["r1v"]:
                    np_["p"] = p["ys"] * s["mult"]
                    np_["oa_r2"] = p["oa_r1"]
                np_["o_we"] = p["r2v"]
                if p["r2v"]:
                    z = rnd(p["p"], s["s1"])
                    np_["o_wa"] = p["oa_r2"]
                    np_["o_wd"] = 0 if (s["relu"] and z < 0) else sat(z, 8)
                # RAMs of the lane (inputs from current registers / combinational)
                self.acc[c].cycle(p["acc_we"], p["acc_wa"], p["acc_wd"], fv and not ffk, m_cur)
                self.bia[c].cycle(0, 0, 0, fv and new_nb, (s["b_base"] + nb_cur) & mb)
                act_wr.append((p["o_we"], p["o_wa"], p["o_wd"] & 0xFF))
                P[c] = np_
            # ---------------- ACT and WGT RAMs (run mode)
            for r in range(n):
                ra_v, ra = ad.tap(r + 1)
                we, wa, wdat = act_wr[r]
                self.act[r].cycle(we, wa, wdat, ra_v, ra)
            old_q = [self.wgt[gi].q for gi in range(ng)]
            for gi in range(ng):
                tv, ta = (s["e_wv"], s["e_wa"]) if gi == 0 else wd.tap(gi * g)
                self.wgt[gi].cycle(0, 0, 0, tv, ta)
            for (gi, l), d in lane_d.items():
                d.step(old_q[gi][l])
            # ---------------- delay lines
            ad.step((s["e_av"], s["e_aa"]))
            av.step(s["e_av"])
            tk.step(s["e_wtok"])
            fl.step((s["e_av"], s["e_fr"], s["e_fk"], s["e_lk"], s["e_fl"]))
            wd.step((s["e_wv"], s["e_wa"]))
            # ---------------- sequencer
            ns = dict(s)
            dq = self.dsc.q
            ns["d_re"] = 0
            ns["e_wv"] = ns["e_av"] = ns["e_wtok"] = 0
            if s["st"] == "LD":
                ld = s["ld"]
                if ld == 1:
                    ns["a_base"], ns["o_base"] = dq & ma, (dq >> 16) & ma
                elif ld == 2:
                    ns["w_tile"] = dq & mw
                elif ld == 3:
                    ns["kb_n"], ns["nb_n"], ns["nb_o"] = dq & 0xFFF, (dq >> 12) & 0xFFF, (dq >> 12) & ma
                elif ld == 4:
                    ns["m_n"], ns["b_base"] = dq & 0xFFF, (dq >> 12) & mb
                elif ld == 5:
                    ns["mult"] = dq & 0x1FFFF
                elif ld == 6:
                    ns["s0"], ns["s1"] = dq & 15, (dq >> 4) & 31
                    ns["relu"], ns["last"] = (dq >> 9) & 1, (dq >> 10) & 1
                if ld <= 4:
                    ns["d_re"], ns["d_ra"] = 1, s["dent"] * 8 + ld + 1
                if ld == 6:
                    ns["kb"] = ns["nb"] = ns["s"] = 0
                    ns["p_n"] = max(s["m_n"], n)
                    ns["st"] = "RUN"
                ns["ld"] = ld + 1
            elif s["st"] == "RUN":
                a_cur = (s["a_base"] + s["kb"]) & ma if s["s"] == 0 else s["a_row"]
                ns["e_wv"] = int(s["s"] < n)
                ns["e_wa"] = (s["w_tile"] + s["s"]) & mw
                ns["e_wtok"] = int(s["s"] == 0)
                ns["e_av"] = int(s["s"] < s["m_n"])
                ns["e_aa"] = a_cur
                ns["e_fr"] = int(s["s"] == 0)
                ns["e_fk"] = int(s["kb"] == 0)
                ns["e_lk"] = int(s["kb"] == s["kb_n"] - 1)
                ns["e_fl"] = int(s["nb"] == 0)
                ns["a_row"] = (a_cur + s["kb_n"]) & ma
                if s["s"] == s["p_n"] - 1:
                    ns["s"] = 0
                    ns["w_tile"] = (s["w_tile"] + n) & mw
                    if s["kb"] == s["kb_n"] - 1:
                        ns["kb"] = 0
                        if s["nb"] == s["nb_n"] - 1:
                            ns["st"], ns["dr"] = "DRAIN", self.drain
                        else:
                            ns["nb"] = s["nb"] + 1
                    else:
                        ns["kb"] = s["kb"] + 1
                else:
                    ns["s"] = s["s"] + 1
            elif s["st"] == "DRAIN":
                ns["dr"] = s["dr"] - 1
                if s["dr"] == 0:
                    if s["last"]:
                        ns["st"], ns["busy"], ns["done"] = "IDLE", 0, 1
                    else:
                        ns["dent"] = s["dent"] + 1
                        ns["ld"] = 0
                        ns["d_re"], ns["d_ra"] = 1, (s["dent"] + 1) * 8
                        ns["st"] = "LD"
            self.dsc.cycle(0, 0, 0, s["d_re"], s["d_ra"])
            W, AO, AV, PS, TK, PP = nW, nAO, nAV, nPS, nTK, nPP
            S = ns
            if S["st"] == "IDLE":
                break
        self.cycles = cyc
        return cyc


# ------------------------------------------------------------------------------- self-check


def random_layer(rng, k, nout, relu, acc_w=32):
    w = [[rng.randint(-128, 127) for _ in range(nout)] for _ in range(k)]
    bias = [rng.randint(-50000, 50000) for _ in range(nout)]
    return Layer(w, bias, rng.randint(0, 6), rng.randint(1, (1 << 17) - 1), rng.randint(14, 24), relu)


def check(n, g, pipe, m, dims, seed):
    rng = random.Random(seed)
    layers = [random_layer(rng, dims[i], dims[i + 1], i < len(dims) - 2) for i in range(len(dims) - 1)]
    x = [[rng.randint(-128, 127) for _ in range(dims[0])] for _ in range(m)]
    img = Image(layers, m, n)
    img.set_input(x)
    sim = CoreSim(n=n, g=g, pipe=pipe)
    sim.load(img)
    cyc = sim.run()
    act = {a: sim.act_lanes(a) for a in img.output_addrs()}
    got = img.read_output(act)
    want = net_ref(layers, x)
    macs = sum(m * ly.k * ly.n for ly in layers)
    ok = got == want
    print("N=%-2d G=%d PIPE=%d M=%-2d dims=%-18s %6d clocks  %5.2f MAC/clock  %s"
          % (n, g, pipe, m, dims, cyc, macs / cyc, "ok" if ok else "MISMATCH"))
    return ok


if __name__ == "__main__":
    oks = [
        check(4, 1, 0, 1, [9, 6], 1),
        check(4, 4, 1, 3, [8, 5, 7], 2),
        check(8, 1, 1, 1, [20, 17, 10], 3),
        check(8, 2, 1, 8, [24, 16, 10], 4),
        check(8, 4, 0, 13, [17, 9, 3], 5),
        check(16, 2, 1, 16, [40, 33, 12], 6),
    ]
    print("all ok" if all(oks) else "FAILED")
