// ttf_array: N x N grid of ttf_pe.
//
// Row r takes lane r of the input vector at the west edge (a_in[8r +: 8], a_v[r]); column c
// takes the weight bus w_bus[8c +: 8] and the load token w_tok[c] at the north edge, and its
// sum leaves at the south edge (ps[PW*c +: PW]). The top row adds to zero.
//
// Timing (cycles relative to the input of a row vector entering lane 0 at t):
//   lane r enters at t + r (the caller skews the lanes), reaches PE(r, c) at t + r + c,
//   column c's sum is valid at t + N + c + PIPE.
// Weights for the tile whose first input enters at t1: the token is at PE(0, c) at
// t1 + c - 1 and moves down a row per clock; the bus carries row r's weight at
// t1 + c + r - 1.

`default_nettype none

module ttf_array #(
    parameter N    = 8,
    parameter PW   = 19,
    parameter PIPE = 0
) (
    input  wire              clk,
    input  wire              rst,
    input  wire [8*N-1:0]    a_in,
    input  wire [N-1:0]      a_v,
    input  wire [8*N-1:0]    w_bus,
    input  wire [N-1:0]      w_tok,
    output wire [PW*N-1:0]   ps
);
    // flattened nets (portable to tools without multi-dimensional net arrays):
    //   ah/avh: into PE(r, c) from the west, index r * (N + 1) + c
    //   psv/tkv: into PE(r, c) from the north, index r * N + c (r = N: the south edge)
    wire [8*N*(N+1)-1:0]  ah;
    wire [N*(N+1)-1:0]    avh;
    wire [PW*N*(N+1)-1:0] psv;
    wire [N*(N+1)-1:0]    tkv;

    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_w
            assign ah[8*(r*(N+1)) +: 8] = a_in[8*r +: 8];
            assign avh[r*(N+1)]         = a_v[r];
        end
        for (c = 0; c < N; c = c + 1) begin : g_n
            assign psv[PW*c +: PW]  = {PW{1'b0}};
            assign tkv[c]           = w_tok[c];
            assign ps[PW*c +: PW]   = psv[PW*(N*N+c) +: PW];
        end
        for (r = 0; r < N; r = r + 1) begin : g_r
            for (c = 0; c < N; c = c + 1) begin : g_c
                ttf_pe #(.PW(PW), .PIPE(PIPE)) u_pe (
                    .clk(clk), .rst(rst),
                    .a_in(ah[8*(r*(N+1)+c) +: 8]), .a_v_in(avh[r*(N+1)+c]),
                    .ps_in(psv[PW*(r*N+c) +: PW]),
                    .w_bus(w_bus[8*c +: 8]), .w_ld(tkv[r*N+c]),
                    .a_out(ah[8*(r*(N+1)+c+1) +: 8]), .a_v_out(avh[r*(N+1)+c+1]),
                    .ps_out(psv[PW*((r+1)*N+c) +: PW]),
                    .w_ld_out(tkv[(r+1)*N+c])
                );
            end
        end
    endgenerate
endmodule

`default_nettype wire
