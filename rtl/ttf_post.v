// ttf_post: the output lane of one array column: accumulation over k-blocks, bias and
// requantization to int8.
//
// The column's sum for row m of a tile arrives one clock after the row's control flags
// (f_*, the "pre" stage), which gives the registered RAMs a clock to read:
//   first k-block (f_fk):  sum = bias + ps          (bias read once, at the tile's row 0)
//   other k-blocks:        sum = acc[m] + ps
//   not the last k-block:  acc[m] <= sum
//   last k-block (f_lk):   y = requant(sum) -> the activation RAM of this lane
// The accumulator RAM is read and written in different clocks for the same row (a tile is
// at least N >= 4 clocks), so no bypass is needed.
//
// requant (bit-exact in model/ttf_model.py):
//   ys = sat18(rnd(sum, s0));  p = ys * mult;  y = sat8(relu ? max(rnd(p, s1), 0) : rnd(p, s1))
//   rnd(x, s) = (x + (2^s >> 1)) >>> s   (round half up; s = 0: x)
// mult is unsigned (17 bits) so ys * mult fits one 18 x 18 multiplier.
//
// Output addresses: element (m, nb * N + c) of the layer's output goes to lane c at
// o_base + m * nb_n + nb; bias of output nb * N + c at b_base + nb in lane c.

`default_nettype none

module ttf_post #(
    parameter PW    = 19,
    parameter ACC_W = 32,
    parameter AW_M  = 6,    // accumulator rows: the largest batch is 2^AW_M
    parameter AW_B  = 8,
    parameter AW_A  = 11
) (
    input  wire              clk,
    input  wire              rst,
    // control, pre stage (one clock before ps)
    input  wire              f_v,
    input  wire              f_fr,   // row 0 of a tile
    input  wire              f_fk,   // first k-block
    input  wire              f_lk,   // last k-block
    input  wire              f_fl,   // first output block of the layer
    // the column sum, main stage
    input  wire [PW-1:0]     ps,
    // layer parameters (stable while the layer runs)
    input  wire [AW_A-1:0]   o_base,
    input  wire [AW_A-1:0]   nb_n,
    input  wire [AW_B-1:0]   b_base,
    input  wire [3:0]        s0,
    input  wire [16:0]       mult,
    input  wire [4:0]        s1,
    input  wire              relu,
    // host writes to the bias RAM
    input  wire              hb_we,
    input  wire [AW_B-1:0]   hb_wa,
    input  wire [ACC_W-1:0]  hb_wd,
    // int8 result to this lane of the activation RAM
    output reg               o_we,
    output reg  [AW_A-1:0]   o_wa,
    output reg  [7:0]        o_wd
);
    // ---------------------------------------------------------------- pre stage
    reg  [AW_M-1:0] m_cnt;
    reg  [AW_B-1:0] nb_cnt;
    reg  [AW_A-1:0] oa_reg;
    wire            new_nb = f_fr & f_fk;
    wire [AW_M-1:0] m_cur  = f_fr ? {AW_M{1'b0}} : m_cnt;
    wire [AW_B-1:0] nb_cur = new_nb ? (f_fl ? {AW_B{1'b0}} : nb_cnt + 1'b1) : nb_cnt;
    wire [AW_A-1:0] nb_ext = nb_cur;    // zero-extended
    wire [AW_A-1:0] oa_cur = f_fr ? o_base + nb_ext : oa_reg + nb_n;

    always @(posedge clk)
        if (f_v) begin
            m_cnt  <= m_cur + 1'b1;
            oa_reg <= oa_cur;
            if (new_nb) nb_cnt <= nb_cur;
        end

    wire [ACC_W-1:0] acc_q, bias_q;
    reg              acc_we;
    reg  [AW_M-1:0]  acc_wa;
    reg  [ACC_W-1:0] acc_wd;

    ttf_ram #(.W(ACC_W), .AW(AW_M)) u_acc (
        .clk(clk), .we(acc_we), .wa(acc_wa), .wd(acc_wd),
        .re(f_v & ~f_fk), .ra(m_cur), .q(acc_q));

    ttf_ram #(.W(ACC_W), .AW(AW_B)) u_bias (
        .clk(clk), .we(hb_we), .wa(hb_wa), .wd(hb_wd),
        .re(f_v & new_nb), .ra(b_base + nb_cur), .q(bias_q));

    reg            v1, fk1, lk1;
    reg [AW_M-1:0] m1;
    reg [AW_A-1:0] oa1;
    always @(posedge clk) begin
        if (rst) v1 <= 1'b0;
        else     v1 <= f_v;
        if (f_v) begin
            fk1 <= f_fk;
            lk1 <= f_lk;
            m1  <= m_cur;
            oa1 <= oa_cur;
        end
    end

    // ---------------------------------------------------------------- main stage
    wire signed [ACC_W-1:0] base = fk1 ? bias_q : acc_q;
    wire signed [ACC_W-1:0] sum  = base + {{(ACC_W-PW){ps[PW-1]}}, ps};

    reg                     r0v;
    reg  signed [ACC_W-1:0] r0;
    reg  [AW_A-1:0]         oa_r0;
    always @(posedge clk) begin
        acc_we <= v1 & ~lk1 & ~rst;
        if (v1 & ~lk1) begin
            acc_wa <= m1;
            acc_wd <= sum;
        end
        r0v <= v1 & lk1 & ~rst;
        if (v1 & lk1) begin
            r0    <= sum;
            oa_r0 <= oa1;
        end
    end
    // acc_we is registered: the write lands one clock after the sum, two clocks after the
    // row's flags; the next k-block's read of the same row comes at least N - 1 >= 1 clocks
    // after that: a tile is at least N >= 4 clocks (ttf_core), so the read sees the write.

    // ---------------------------------------------------------------- requant
    // R1: ys = sat18(rnd(r0, s0))
    wire signed [ACC_W:0] h0  = {{(ACC_W){1'b0}}, 1'b1} << s0;
    wire signed [ACC_W:0] x1  = {r0[ACC_W-1], r0} + (h0 >>> 1);
    wire signed [ACC_W:0] x2  = x1 >>> s0;
    wire                  x2_hi = (x2 > 131071);
    wire                  x2_lo = (x2 < -131072);
    reg                   r1v;
    reg  signed [17:0]    ys;
    reg  [AW_A-1:0]       oa_r1;
    always @(posedge clk) begin
        r1v <= r0v & ~rst;
        if (r0v) begin
            ys    <= x2_hi ? 18'sh1FFFF : x2_lo ? 18'sh20000 : x2[17:0];
            oa_r1 <= oa_r0;
        end
    end

    // R2: p = ys * mult
    reg                r2v;
    reg signed [35:0]  p;
    reg [AW_A-1:0]     oa_r2;
    always @(posedge clk) begin
        r2v <= r1v & ~rst;
        if (r1v) begin
            p     <= ys * $signed({1'b0, mult});
            oa_r2 <= oa_r1;
        end
    end

    // R3: y = sat8(relu(rnd(p, s1)))
    wire signed [36:0] g0  = {{36{1'b0}}, 1'b1} << s1;
    wire signed [36:0] z1  = {p[35], p} + (g0 >>> 1);
    wire signed [36:0] z2  = z1 >>> s1;
    wire               neg = z2[36];
    wire               z_hi = ~neg & (z2 > 37'sd127);
    wire               z_lo = neg & (z2 < -37'sd128);
    always @(posedge clk) begin
        o_we <= r2v & ~rst;
        if (r2v) begin
            o_wa <= oa_r2;
            o_wd <= (relu & neg) ? 8'd0 : z_hi ? 8'd127 : z_lo ? 8'h80 : z2[7:0];
        end
    end
endmodule

`default_nettype wire
