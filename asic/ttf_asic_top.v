// ttf_asic_top: ttf_core sized for a first OpenLane run on sky130 (asic/config.json).
//
// The RAMs are ttf_ram, which synthesis turns into flip-flops: the sizes here are kept
// small (256 activation rows, 256 weight rows, batch 16) so the flow measures the array,
// the lanes and the control rather than a sea of RAM flops. A real chip replaces ttf_ram
// with SRAM macros (PLAN.md, phase 3); the core's ports and timing stay the same.

`default_nettype none

module ttf_asic_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        h_we,
    input  wire        h_re,
    input  wire [15:0] h_a,
    input  wire [31:0] h_wd,
    output wire [31:0] h_rd,
    output wire        h_rv,
    output wire        busy,
    output wire        done
);
    ttf_core #(.N(8), .G(1), .PIPE(1), .ACC_W(24), .AW_A(8), .AW_W(8), .AW_M(4), .AW_B(4),
               .AW_D(5)) u_core (
        .clk(clk), .rst(rst), .h_we(h_we), .h_re(h_re), .h_a(h_a), .h_wd(h_wd),
        .h_rd(h_rd), .h_rv(h_rv), .busy(busy), .done(done));
endmodule

`default_nettype wire
