// ttf_pe: one processing element of the weight-stationary int8 array.
//
// a (int8) enters from the west and leaves east one clock later; the partial sum enters
// from the north and leaves south one clock later (PIPE = 0) as ps_in + a * w.
//
// The weight has no shadow copy: it is written in place, w <= w_bus when w_ld, on the
// clock edge just before the first input of the next tile reaches this PE. The load token
// (w_ld) travels down the column one row per clock, the weight bus is shared by the column
// and carries row r's weight while the token is at row r (ttf_core skews both per column).
//
// Registers only change on valid data: nothing toggles while the array is idle. Only the
// two control flags are reset.
//
// PIPE = 1 registers the product first (one more clock of latency, a shorter path):
// ps_out = ps_in + p, with p the product of the input one clock earlier. The north
// neighbour's sum arrives one clock after its own input, which is the clock this PE's
// product is ready, so the column still adds up the right terms.

`default_nettype none

module ttf_pe #(
    parameter PW   = 19,    // partial-sum width: 16 + log2(N) holds N int8 products exactly
    parameter PIPE = 0
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire signed [7:0]    a_in,
    input  wire                 a_v_in,
    input  wire signed [PW-1:0] ps_in,
    input  wire [7:0]           w_bus,
    input  wire                 w_ld,
    output reg  signed [7:0]    a_out,
    output reg                  a_v_out,
    output reg  signed [PW-1:0] ps_out,
    output reg                  w_ld_out
);
    reg signed [7:0] w;
    always @(posedge clk)
        if (w_ld) w <= w_bus;

    always @(posedge clk)
        if (rst) begin
            a_v_out  <= 1'b0;
            w_ld_out <= 1'b0;
        end else begin
            a_v_out  <= a_v_in;
            w_ld_out <= w_ld;
        end

    always @(posedge clk)
        if (a_v_in) a_out <= a_in;

    wire signed [15:0] prod = a_in * w;

    generate
        if (PIPE) begin : g_pipe
            reg signed [15:0] p;
            always @(posedge clk)
                if (a_v_in) p <= prod;
            always @(posedge clk)
                if (a_v_out) ps_out <= ps_in + {{(PW-16){p[15]}}, p};
        end else begin : g_comb
            always @(posedge clk)
                if (a_v_in) ps_out <= ps_in + {{(PW-16){prod[15]}}, prod};
        end
    endgenerate
endmodule

`default_nettype wire
