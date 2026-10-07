// ttf_dly: a shift register of L stages; tap k (1..L) is the input delayed k clocks.
// taps[W*(k-1) +: W] = in delayed k clocks.

`default_nettype none

module ttf_dly #(
    parameter W = 1,
    parameter L = 1
) (
    input  wire         clk,
    input  wire [W-1:0] in,
    output wire [W*L-1:0] taps
);
    reg [W*L-1:0] sr;
    generate
        if (L == 1) begin : g1
            always @(posedge clk) sr <= in;
        end else begin : gn
            always @(posedge clk) sr <= {sr[W*(L-1)-1:0], in};
        end
    endgenerate
    assign taps = sr;
endmodule

`default_nettype wire
