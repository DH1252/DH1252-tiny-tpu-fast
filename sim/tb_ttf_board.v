// tb_ttf_board: the Tang Nano 20K top (boards/tn20k/ttf_tn20k_top.v) driven byte by byte
// over its UART, as tools/ttf_host.py does: ping, burst writes of the image, run, poll
// STATUS, read the outputs. Covers what tb_ttf_core leaves out: the UART bridge, the
// burst address counter, the reply FIFO and the core clock gate (sim/ttf_dqce_sim.v).
//
//   make -f fast.mk sim-board [RANDOM=1 M=4]        (the MNIST image takes minutes)
//
// The UART runs at CLK_HZ / BAUD = 8 clocks per bit to keep the run short.

`timescale 1ns/1ps
`default_nettype none

module tb_ttf_board;
    parameter CLKGATE = 1;
    parameter DIR     = "build/sim";
    parameter MAXW    = 65536;
    localparam DIV    = 8;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg  rx = 1'b1;            // PC -> board
    wire tx;                   // board -> PC
    wire [5:0] led_n;

    ttf_tn20k_top #(.CLK_HZ(DIV * 1000000), .BAUD(1000000), .CLKGATE(CLKGATE)) dut (
        .clk(clk), .clk_ok(1'b1), .uart_rx(rx), .uart_tx(tx), .led_n(led_n));

    // ---------------------------------------------------------------- PC side of the UART
    task send(input [7:0] b);
        integer k;
        begin
            rx = 1'b0; repeat (DIV) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin rx = b[k]; repeat (DIV) @(posedge clk); end
            rx = 1'b1; repeat (DIV) @(posedge clk);
        end
    endtask

    reg [7:0] q [0:1023];
    integer   q_wp = 0, q_rp = 0;
    integer   k2;
    reg [7:0] sh;
    always begin
        @(negedge tx);
        repeat (DIV / 2) @(posedge clk);
        for (k2 = 0; k2 < 8; k2 = k2 + 1) begin
            repeat (DIV) @(posedge clk);
            sh[k2] = tx;
        end
        repeat (DIV) @(posedge clk);
        q[q_wp % 1024] = sh;
        q_wp = q_wp + 1;
    end

    task recv(output [7:0] b);
        integer t;
        begin
            t = 0;
            while (q_rp == q_wp) begin
                @(posedge clk);
                t = t + 1;
                if (t > 200000) begin $display("FAIL: no reply from the board"); $finish; end
            end
            b = q[q_rp % 1024];
            q_rp = q_rp + 1;
        end
    endtask

    task expect_k;
        reg [7:0] b;
        begin
            recv(b);
            if (b !== "K") begin $display("FAIL: reply %h, not 'K'", b); $finish; end
        end
    endtask

    task rd(input [15:0] a, output [31:0] d);
        reg [7:0] b0, b1, b2, b3;
        begin
            send("R"); send(a[7:0]); send(a[15:8]);
            recv(b0); recv(b1); recv(b2); recv(b3);
            d = {b3, b2, b1, b0};
        end
    endtask

    task wr(input [15:0] a, input [31:0] d);
        begin
            send("W"); send(a[7:0]); send(a[15:8]);
            send(d[7:0]); send(d[15:8]); send(d[23:16]); send(d[31:24]);
            expect_k;
        end
    endtask

    // ---------------------------------------------------------------- the test
    reg [47:0] wl [0:MAXW-1];
    reg [47:0] ex [0:MAXW-1];
    integer i, j, n, nw, errs, nexp;
    reg [31:0] d;
    reg [7:0]  b;

    initial begin
        $readmemh({DIR, "/writes.memh"}, wl);
        $readmemh({DIR, "/expect.memh"}, ex);
        nw = 0;
        while (wl[nw][47:32] !== 16'hFFFF) nw = nw + 1;
        repeat (40000) @(posedge clk);                 // the board's power-on reset
        send("P"); expect_k;
        rd(16'hF000, d);
        if (d !== 32'h54544631) begin $display("FAIL: ID %h", d); $finish; end
        // the image as bursts of consecutive addresses, up to 256 words each
        i = 0;
        while (i < nw) begin
            n = 1;
            while (i + n < nw && n < 256 && wl[i + n][47:32] == wl[i][47:32] + n) n = n + 1;
            send("B"); send(wl[i][39:32]); send(wl[i][47:40]); send(n[7:0]);
            for (j = 0; j < n; j = j + 1) begin
                send(wl[i + j][7:0]); send(wl[i + j][15:8]); send(wl[i + j][23:16]); send(wl[i + j][31:24]);
            end
            expect_k;
            i = i + n;
        end
        $display("tb_ttf_board: %0d words written over the UART", nw);
        // inputs read back through the bridge
        errs = 0;
        for (i = 0; i < nw; i = i + 1)
            if (wl[i][47:46] == 2'b10) begin
                rd(wl[i][47:32], d);
                if (d !== wl[i][31:0]) errs = errs + 1;
            end
        $display("tb_ttf_board: input read-back, %0d words differ", errs);
        wr(16'hF001, 32'd1);
        d = 0;
        while (!d[1]) rd(16'hF002, d);
        rd(16'hF003, d);
        $display("tb_ttf_board: CYCLES %0d", d);
        for (nexp = 0; ex[nexp][47:32] !== 16'hFFFF; nexp = nexp + 1) begin
            rd(ex[nexp][47:32], d);
            if (d !== ex[nexp][31:0]) begin
                errs = errs + 1;
                if (errs <= 10) $display("  %h: got %h want %h", ex[nexp][47:32], d, ex[nexp][31:0]);
            end
        end
        if (errs == 0) $display("PASS: %0d output words (CLKGATE=%0d)", nexp, CLKGATE);
        else           $display("FAIL: %0d words differ (CLKGATE=%0d)", errs, CLKGATE);
        $finish;
    end
endmodule

`default_nettype wire
