// ttf_ram: simple dual-port RAM, one write port and one read port, registered read.
//
// q holds the last word read while re is low, so a value read once can be used for many
// clocks (the bias of a tile) and an idle read port does not toggle. Written so that Gowin
// and Yosys infer block RAM; for the ASIC it is the place to substitute a RAM macro
// (OpenRAM, DFFRAM) with the same ports.

`default_nettype none

module ttf_ram #(
    parameter W  = 8,
    parameter AW = 10,
    parameter D  = 1 << AW
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] wa,
    input  wire [W-1:0]  wd,
    input  wire          re,
    input  wire [AW-1:0] ra,
    output reg  [W-1:0]  q
);
    reg [W-1:0] mem [0:D-1];
    always @(posedge clk) begin
        if (we) mem[wa] <= wd;
        if (re) q <= mem[ra];
    end
endmodule

`default_nettype wire
