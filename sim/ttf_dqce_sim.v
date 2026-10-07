// ttf_dqce_sim: a stand-in for Gowin's DQCE clock buffer in simulation (CLKGATE = 1).
// The enable is taken while the clock is low, so the output never has a short pulse.
`timescale 1ns/1ps
module DQCE (input wire CLKIN, input wire CE, output wire CLKOUT);
    reg en = 1'b0;
    always @(CLKIN or CE)
        if (!CLKIN) en = CE;
    assign CLKOUT = CLKIN & en;
endmodule
