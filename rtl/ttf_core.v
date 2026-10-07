// ttf_core: the int8 inference core. Array, memories, output lanes, sequencer, host bus.
//
// Memories, all ttf_ram (one write port, one registered read port):
//   ACT  N lanes x 2^AW_A x 8 bit   activations; lane r holds element kb * N + r of each
//                                   row block; read by the array, written by the lanes
//   WGT  N/G RAMs x 2^AW_W x 8G bit weights; row address = tile * N + k, lane = column
//   ACC, BIAS                        in each output lane (ttf_post)
//   DESC 2^AW_D x 32 bit            layer descriptors (ttf_seq)
// Each lane has its own RAM and its own (delayed) address, so the lanes need no data skew
// registers: lane r of the input is read r clocks after lane 0, column c's weight bus is
// fed c clocks after column 0, and output lane c writes when its column's sum is ready.
// G > 1 packs G weight lanes into one RAM word (fewer, wider RAMs: better block RAM use on
// an FPGA) at the price of G (G - 1) / 2 byte registers per group to skew its lanes.
//
// Host bus (word addresses, 32-bit data; the RAMs only while no program runs):
//   0x0000-0x7FFF  WGT   row w, lanes 4q .. 4q+3 at w * N/4 + q        (write)
//   0x8000-0xBFFF  ACT   row a, lanes 4q .. 4q+3 at 0x8000 + a * N/4 + q (write, read)
//   0xC000-0xDFFF  BIAS  row b, lane c at 0xC000 + b * N + c            (write)
//   0xE000-0xEFFF  DESC  word i at 0xE000 + i                          (write)
//   0xF000 ID "TTF1"   0xF001 CTRL (write 1: run the program)   0xF002 STATUS (busy, done)
//   0xF003 CYCLES of the last run   0xF004 PARAMS   0xF005 SIZES
// A read returns its data on h_rd in the clock after h_re (h_rv).

`default_nettype none

module ttf_core #(
    parameter N     = 8,     // array size: 4, 8 or 16
    parameter G     = 1,     // weight lanes per WGT RAM: 1, 2 or 4
    parameter PIPE  = 1,
    parameter ACC_W = 32,
    parameter AW_A  = 11,
    parameter AW_W  = 13,
    parameter W_D   = 1 << AW_W,   // WGT rows actually built (block RAM budget)
    parameter AW_M  = 6,
    parameter AW_B  = 8,
    parameter AW_D  = 6
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        h_we,
    input  wire        h_re,
    input  wire [15:0] h_a,
    input  wire [31:0] h_wd,
    output wire [31:0] h_rd,
    output reg         h_rv,
    output wire        busy,
    output wire        done
);
    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2 = clog2 + 1;
        end
    endfunction

    localparam LOGN  = clog2(N);
    localparam PW    = 16 + LOGN;
    localparam QB    = LOGN - 2;            // host words per row: N / 4 = 2^QB
    localparam NG    = N / G;
    localparam OUT_D = N + 2 + PIPE;        // emission to column 0's sum
    localparam DRAIN = OUT_D + N + 8;
    localparam [7:0] P_N = N, P_G = G, P_ACC = ACC_W;
    localparam [4:0] P_A = AW_A, P_W = AW_W, P_M = AW_M, P_B = AW_B, P_D = AW_D;
    localparam       P_P = PIPE ? 1'b1 : 1'b0;

    // ---------------------------------------------------------------- host decode
    wire r_wgt  = ~h_a[15];
    wire r_act  = h_a[15:14] == 2'b10;
    wire r_bias = h_a[15:13] == 3'b110;
    wire r_desc = h_a[15:12] == 4'hE;
    wire r_reg  = h_a[15:12] == 4'hF;
    wire [14:0] h_row_w = h_a[14:0] >> QB;
    wire [13:0] h_row_a = h_a[13:0] >> QB;
    wire [3:0]  h_q     = h_a[3:0] & ((1 << QB) - 1);
    wire [12:0] h_row_b = h_a[12:0] >> LOGN;
    wire [4:0]  h_c     = h_a[4:0] & (N - 1);
    wire        hw      = h_we & ~busy;

    wire start = h_we & r_reg & (h_a[3:0] == 4'h1) & h_wd[0] & ~busy;

    // ---------------------------------------------------------------- sequencer
    wire            d_re;
    wire [AW_D-1:0] d_ra;
    wire [31:0]     d_q;
    wire            e_wv, e_wtok, e_av, e_fr, e_fk, e_lk, e_fl;
    wire [AW_W-1:0] e_wa;
    wire [AW_A-1:0] e_aa;
    wire [AW_A-1:0] o_base, nb_o;
    wire [AW_B-1:0] b_base;
    wire [3:0]      s0;
    wire [16:0]     mult;
    wire [4:0]      s1;
    wire            relu;

    ttf_seq #(.N(N), .AW_A(AW_A), .AW_W(AW_W), .AW_B(AW_B), .AW_D(AW_D), .DRAIN(DRAIN)) u_seq (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .d_re(d_re), .d_ra(d_ra), .d_q(d_q),
        .e_wv(e_wv), .e_wa(e_wa), .e_wtok(e_wtok),
        .e_av(e_av), .e_aa(e_aa), .e_fr(e_fr), .e_fk(e_fk), .e_lk(e_lk), .e_fl(e_fl),
        .o_base(o_base), .nb_o(nb_o), .b_base(b_base),
        .s0(s0), .mult(mult), .s1(s1), .relu(relu));

    ttf_ram #(.W(32), .AW(AW_D)) u_desc (
        .clk(clk), .we(hw & r_desc), .wa(h_a[AW_D-1:0]), .wd(h_wd),
        .re(d_re), .ra(d_ra), .q(d_q));

    // ---------------------------------------------------------------- delay lines
    // ACT lane r reads at emission + r + 1; its data enters the array at emission + r + 2
    wire [(AW_A+1)*N-1:0] ad_t;
    wire [N:0]            av_t;
    ttf_dly #(.W(AW_A + 1), .L(N)) u_ad (.clk(clk), .in({e_av, e_aa}), .taps(ad_t));
    ttf_dly #(.W(1), .L(N + 1))    u_av (.clk(clk), .in(e_av), .taps(av_t));
    // column c's load token at emission + c + 1 (weight RAM read at emission, group skew,
    // then lane skew inside the group: column c's bus is valid at emission + c + 1)
    wire [N-1:0] tk_t;
    ttf_dly #(.W(1), .L(N)) u_tk (.clk(clk), .in(e_wtok), .taps(tk_t));
    // the output lanes' flags: column c's pre stage at emission + OUT_D + c - 1
    localparam FL = OUT_D + N - 1;
    wire [5*FL-1:0] fl_t;
    ttf_dly #(.W(5), .L(FL)) u_fl (.clk(clk), .in({e_av, e_fr, e_fk, e_lk, e_fl}), .taps(fl_t));

    // ---------------------------------------------------------------- ACT lanes
    wire [8*N-1:0] act_q;
    wire [N-1:0]   o_we;
    wire [AW_A*N-1:0] o_wa;
    wire [8*N-1:0] o_wd;

    genvar r, c, g, l;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_act
            wire [AW_A:0] t = ad_t[(AW_A+1)*r +: AW_A+1];   // tap r + 1
            ttf_ram #(.W(8), .AW(AW_A)) u_ram (
                .clk(clk),
                .we(busy ? o_we[r] : hw & r_act & ((r >> 2) == h_q)),
                .wa(busy ? o_wa[AW_A*r +: AW_A] : h_row_a[AW_A-1:0]),
                .wd(busy ? o_wd[8*r +: 8] : h_wd[8*(r & 3) +: 8]),
                .re(busy ? t[AW_A] : h_re & r_act),
                .ra(busy ? t[AW_A-1:0] : h_row_a[AW_A-1:0]),
                .q(act_q[8*r +: 8]));
        end
    endgenerate

    // ---------------------------------------------------------------- WGT RAMs
    wire [8*N-1:0] w_bus;
    generate
        if (NG > 1) begin : g_wd
            wire [(AW_W+1)*(N-G)-1:0] wd_t;
            ttf_dly #(.W(AW_W + 1), .L(N - G)) u_wd (.clk(clk), .in({e_wv, e_wa}), .taps(wd_t));
            for (g = 0; g < NG; g = g + 1) begin : g_grp
                wire [AW_W:0] t;                 // tap g * G
                if (g == 0) begin : g_t0
                    assign t = {e_wv, e_wa};
                end else begin : g_tn
                    assign t = wd_t[(AW_W+1)*(g*G-1) +: AW_W+1];
                end
                wire [8*G-1:0] q;
                ttf_ram #(.W(8 * G), .AW(AW_W), .D(W_D)) u_ram (
                    .clk(clk),
                    .we(hw & r_wgt & (((g * G) >> 2) == h_q)),
                    .wa(h_row_w[AW_W-1:0]),
                    .wd(h_wd[8*((g*G) & 3) +: 8*G]),
                    .re(t[AW_W]), .ra(t[AW_W-1:0]), .q(q));
                for (l = 0; l < G; l = l + 1) begin : g_lane
                    if (l == 0) begin : g0
                        assign w_bus[8*(g*G) +: 8] = q[7:0];
                    end else begin : gl
                        wire [8*l-1:0] dt;
                        ttf_dly #(.W(8), .L(l)) u_ld (.clk(clk), .in(q[8*l +: 8]), .taps(dt));
                        assign w_bus[8*(g*G+l) +: 8] = dt[8*(l-1) +: 8];
                    end
                end
            end
        end else begin : g_w1
            wire [8*N-1:0] q;
            ttf_ram #(.W(8 * N), .AW(AW_W), .D(W_D)) u_ram (
                .clk(clk), .we(hw & r_wgt), .wa(h_row_w[AW_W-1:0]), .wd(h_wd[8*N-1:0]),
                .re(e_wv), .ra(e_wa), .q(q));
            for (l = 0; l < N; l = l + 1) begin : g_lane
                if (l == 0) begin : g0
                    assign w_bus[7:0] = q[7:0];
                end else begin : gl
                    wire [8*l-1:0] dt;
                    ttf_dly #(.W(8), .L(l)) u_ld (.clk(clk), .in(q[8*l +: 8]), .taps(dt));
                    assign w_bus[8*l +: 8] = dt[8*(l-1) +: 8];
                end
            end
        end
    endgenerate

    // ---------------------------------------------------------------- array
    wire [N-1:0]    a_v;
    wire [PW*N-1:0] ps;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_av
            assign a_v[r] = av_t[r + 1];                    // tap r + 2
        end
    endgenerate

    ttf_array #(.N(N), .PW(PW), .PIPE(PIPE)) u_arr (
        .clk(clk), .rst(rst), .a_in(act_q), .a_v(a_v), .w_bus(w_bus), .w_tok(tk_t), .ps(ps));

    // ---------------------------------------------------------------- output lanes
    generate
        for (c = 0; c < N; c = c + 1) begin : g_post
            wire [4:0] f = fl_t[5*(OUT_D + c - 2) +: 5];    // tap OUT_D + c - 1
            ttf_post #(.PW(PW), .ACC_W(ACC_W), .AW_M(AW_M), .AW_B(AW_B), .AW_A(AW_A)) u_post (
                .clk(clk), .rst(rst),
                .f_v(f[4]), .f_fr(f[3]), .f_fk(f[2]), .f_lk(f[1]), .f_fl(f[0]),
                .ps(ps[PW*c +: PW]),
                .o_base(o_base), .nb_n(nb_o), .b_base(b_base),
                .s0(s0), .mult(mult), .s1(s1), .relu(relu),
                .hb_we(hw & r_bias & (h_c == c)), .hb_wa(h_row_b[AW_B-1:0]),
                .hb_wd({{(ACC_W > 32 ? ACC_W - 32 : 1){h_wd[31]}}, h_wd} ),
                .o_we(o_we[c]), .o_wa(o_wa[AW_A*c +: AW_A]), .o_wd(o_wd[8*c +: 8]));
        end
    endgenerate

    // ---------------------------------------------------------------- registers, reads
    reg  [31:0] cycles;
    always @(posedge clk)
        if (rst)        cycles <= 32'd0;
        else if (start) cycles <= 32'd0;
        else if (busy)  cycles <= cycles + 1'b1;

    reg        rd_act;
    reg  [3:0] rd_q;
    reg [31:0] rd_reg;
    always @(posedge clk) begin
        h_rv   <= h_re;
        rd_act <= h_re & r_act;
        rd_q   <= h_q;
        case (h_a[3:0])
        4'h0:    rd_reg <= 32'h54544631;                    // "TTF1"
        4'h2:    rd_reg <= {30'd0, done, busy};
        4'h3:    rd_reg <= cycles;
        4'h4:    rd_reg <= {7'd0, P_P, P_ACC, P_G, P_N};
        4'h5:    rd_reg <= {7'd0, P_D, P_B, P_M, P_W, P_A};
        default: rd_reg <= 32'd0;
        endcase
    end
    wire [31:0] act_word = act_q[32*rd_q +: 32];
    assign h_rd = rd_act ? act_word : rd_reg;
endmodule

`default_nettype wire
