// tb_ttf_core: loads a network through the host bus, runs it, checks the outputs.
//
//   python3 tools/ttf_image.py --out build/sim [--m 8 | --random 1]
//   make sim            (iverilog; or VERILATOR=1 make sim)
//
// writes.memh / expect.memh: "aaaadddddddd" per line, ending with ffff00000000 (tools/ttf_image.py).
// Prints the program's clocks (CYCLES) and PASS or FAIL.

`timescale 1ns/1ps
`default_nettype none

module tb_ttf_core;
    parameter N     = 8;
    parameter G     = 4;
    parameter PIPE  = 1;
    parameter ACC_W = 24;
    parameter AW_A  = 11;
    parameter AW_W  = 13;
    parameter W_D   = 6400;
    parameter AW_M  = 4;
    parameter AW_B  = 4;
    parameter AW_D  = 6;
    parameter DIR   = "build/sim";
    parameter MAXW  = 65536;

    reg clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;

    reg         h_we = 1'b0, h_re = 1'b0;
    reg  [15:0] h_a = 16'd0;
    reg  [31:0] h_wd = 32'd0;
    wire [31:0] h_rd;
    wire        h_rv, busy, done;

    ttf_core #(.N(N), .G(G), .PIPE(PIPE), .ACC_W(ACC_W), .AW_A(AW_A), .AW_W(AW_W), .W_D(W_D),
               .AW_M(AW_M), .AW_B(AW_B), .AW_D(AW_D)) dut (
        .clk(clk), .rst(rst), .h_we(h_we), .h_re(h_re), .h_a(h_a), .h_wd(h_wd),
        .h_rd(h_rd), .h_rv(h_rv), .busy(busy), .done(done));

    reg [47:0] wr [0:MAXW-1];
    reg [47:0] ex [0:MAXW-1];

    task bus_write(input [15:0] a, input [31:0] d);
        begin
            @(negedge clk); h_we = 1'b1; h_a = a; h_wd = d;
            @(negedge clk); h_we = 1'b0;
        end
    endtask

    reg [31:0] rdata;
    task bus_read(input [15:0] a);
        begin
            @(negedge clk); h_re = 1'b1; h_a = a;
            @(negedge clk); h_re = 1'b0;
            rdata = h_rd;                 // h_rv is high in this clock
        end
    endtask

    integer i, errs, nexp;
    initial begin
        $readmemh({DIR, "/writes.memh"}, wr);
        $readmemh({DIR, "/expect.memh"}, ex);
        repeat (4) @(posedge clk);
        rst = 1'b0;
        bus_read(16'hF000);
        if (rdata !== 32'h54544631) begin $display("FAIL: ID %h", rdata); $finish; end
        for (i = 0; wr[i][47:32] !== 16'hFFFF; i = i + 1)
            bus_write(wr[i][47:32], wr[i][31:0]);
        $display("tb_ttf_core: %0d words loaded, running", i);
        bus_write(16'hF001, 32'd1);
        @(posedge done);
        bus_read(16'hF003);
        $display("tb_ttf_core: CYCLES %0d", rdata);
        errs = 0;
        for (nexp = 0; ex[nexp][47:32] !== 16'hFFFF; nexp = nexp + 1) begin
            bus_read(ex[nexp][47:32]);
            if (rdata !== ex[nexp][31:0]) begin
                errs = errs + 1;
                if (errs <= 10)
                    $display("  %h: got %h want %h", ex[nexp][47:32], rdata, ex[nexp][31:0]);
            end
        end
        if (errs == 0) $display("PASS: %0d output words", nexp);
        else           $display("FAIL: %0d of %0d output words differ", errs, nexp);
        $finish;
    end

    initial begin
        #500000000;
        $display("FAIL: timeout");
        $finish;
    end
endmodule

`default_nettype wire
