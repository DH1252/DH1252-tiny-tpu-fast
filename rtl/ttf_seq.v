// ttf_seq: runs a program of layer descriptors and emits one tile slot item per clock.
//
// A layer computes out[m][n] = requant(sum_k in[m][k] * w[k][n] + bias[n]) for m < m_n
// (the batch), in tiles of N x N weights: for nb < nb_n (output blocks), for kb < kb_n
// (input blocks), one tile slot of P = max(m_n, N) clocks. Clock s of a slot emits
//   - weight row s of the tile (s < N): WGT address w_base + (nb * kb_n + kb) * N + s,
//     with the load token on s = 0;
//   - input row m = s (s < m_n): ACT address a_base + m * kb_n + kb, and the output lanes'
//     flags (row 0, first / last k-block, first output block).
// ttf_core delays these per lane so the weights land in the array just in time for the
// tile's first input, while the previous tile is still running (no stall, no shadow
// weights): with m_n >= N the array does a full N x N MACs every clock.
//
// Descriptor (8 words of 32 bits, entry d at DESC words 8d .. 8d + 7):
//   0: a_base [15:0], o_base [31:16]      (ACT addresses)
//   1: w_base [23:0]                      (WGT row address)
//   2: kb_n [11:0], nb_n [23:12]          (blocks, >= 1)
//   3: m_n [11:0], b_base [27:12]         (rows 1 .. 2^AW_M, bias row address)
//   4: mult [16:0]
//   5: s0 [3:0], s1 [8:4], relu [9], last [10]
// The program starts at entry 0 and ends after the entry with last set. Between layers the
// pipeline drains, so a layer may read what the one before it wrote.

`default_nettype none

module ttf_seq #(
    parameter N     = 8,
    parameter AW_A  = 11,
    parameter AW_W  = 13,
    parameter AW_B  = 8,
    parameter AW_D  = 6,
    parameter DRAIN = 32
) (
    input  wire            clk,
    input  wire            rst,
    input  wire            start,
    output reg             busy,
    output reg             done,
    // descriptor RAM read port
    output reg             d_re,
    output reg  [AW_D-1:0] d_ra,
    input  wire [31:0]     d_q,
    // emission (one clock)
    output reg             e_wv,     // weight row valid
    output reg  [AW_W-1:0] e_wa,
    output reg             e_wtok,   // weight row 0 of a tile
    output reg             e_av,     // input row valid
    output reg  [AW_A-1:0] e_aa,
    output reg             e_fr,     // input row 0 of a tile
    output reg             e_fk,
    output reg             e_lk,
    output reg             e_fl,
    // layer parameters for the output lanes
    output reg  [AW_A-1:0] o_base,
    output reg  [AW_A-1:0] nb_o,
    output reg  [AW_B-1:0] b_base,
    output reg  [3:0]      s0,
    output reg  [16:0]     mult,
    output reg  [4:0]      s1,
    output reg             relu
);
    localparam S_IDLE = 2'd0, S_LD = 2'd1, S_RUN = 2'd2, S_DRAIN = 2'd3;
    reg [1:0]  st;
    reg [2:0]  ld;                 // descriptor word being loaded
    reg [AW_D-4:0] dent;           // descriptor entry
    reg        last;
    reg [AW_A-1:0] a_base;
    reg [AW_W-1:0] w_tile;
    reg [11:0] kb_n, nb_n, m_n;
    reg [11:0] kb, nb, s, p_n;
    reg [AW_A-1:0] a_row;
    reg [7:0]  dr;

    wire [AW_A-1:0] kb_a = kb;     // widths adjusted for the address adders
    wire [AW_A-1:0] kbn_a = kb_n;
    wire [AW_W-1:0] s_w = s;
    wire [AW_A-1:0] a_cur = (s == 0) ? a_base + kb_a : a_row;
    wire            last_s  = (s == p_n - 1'b1);
    wire            last_kb = (kb == kb_n - 1'b1);
    wire            last_nb = (nb == nb_n - 1'b1);

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; d_re <= 1'b0;
            e_wv <= 1'b0; e_av <= 1'b0; e_wtok <= 1'b0;
        end else begin
            d_re   <= 1'b0;
            e_wv   <= 1'b0;
            e_av   <= 1'b0;
            e_wtok <= 1'b0;
            case (st)
            S_IDLE:
                if (start) begin
                    busy <= 1'b1; done <= 1'b0;
                    dent <= 0; ld <= 3'd0;
                    d_re <= 1'b1; d_ra <= 0;
                    st <= S_LD;
                end
            S_LD: begin
                // word k is on d_ra during ld = k (registered the clock before) and on d_q
                // during ld = k + 1
                case (ld)
                3'd1: begin a_base <= d_q[AW_A-1:0]; o_base <= d_q[16 +: AW_A]; end
                3'd2: w_tile <= d_q[AW_W-1:0];
                3'd3: begin kb_n <= d_q[11:0]; nb_n <= d_q[23:12]; nb_o <= d_q[12 +: AW_A]; end
                3'd4: begin m_n <= d_q[11:0]; b_base <= d_q[12 +: AW_B]; end
                3'd5: mult <= d_q[16:0];
                3'd6: begin s0 <= d_q[3:0]; s1 <= d_q[8:4]; relu <= d_q[9]; last <= d_q[10]; end
                default: ;
                endcase
                if (ld <= 3'd4) begin
                    d_re <= 1'b1;
                    d_ra <= {dent, ld + 3'd1};
                end
                if (ld == 3'd6) begin
                    kb <= 0; nb <= 0; s <= 0;
                    p_n <= (m_n > N) ? m_n : N;
                    st <= S_RUN;
                end
                ld <= ld + 1'b1;
            end
            S_RUN: begin
                e_wv   <= (s < N);
                e_wa   <= w_tile + s_w;
                e_wtok <= (s == 0);
                e_av   <= (s < m_n);
                e_aa   <= a_cur;
                e_fr   <= (s == 0);
                e_fk   <= (kb == 0);
                e_lk   <= last_kb;
                e_fl   <= (nb == 0);
                a_row  <= a_cur + kbn_a;
                if (last_s) begin
                    s      <= 0;
                    w_tile <= w_tile + N;
                    if (last_kb) begin
                        kb <= 0;
                        if (last_nb) begin
                            st <= S_DRAIN; dr <= DRAIN;
                        end else
                            nb <= nb + 1'b1;
                    end else
                        kb <= kb + 1'b1;
                end else
                    s <= s + 1'b1;
            end
            default: begin // S_DRAIN
                dr <= dr - 1'b1;
                if (dr == 0) begin
                    if (last) begin
                        st <= S_IDLE; busy <= 1'b0; done <= 1'b1;
                    end else begin
                        dent <= dent + 1'b1; ld <= 3'd0;
                        d_re <= 1'b1; d_ra <= {dent + 1'b1, 3'd0};
                        st <= S_LD;
                    end
                end
            end
            endcase
        end
    end
endmodule

`default_nettype wire
