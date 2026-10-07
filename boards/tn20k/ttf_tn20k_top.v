// ttf_tn20k_top: the int8 inference core on the Sipeed Tang Nano 20K, driven from a PC
// over the BL616's USB-to-UART bridge.
//
// Byte protocol (8N1 at BAUD; addresses and data little-endian):
//   'P'                          -> 'K'                 ping
//   'W' a0 a1 d0 d1 d2 d3        -> 'K'                 write a 32-bit word
//   'B' a0 a1 n  (4 n data bytes) -> 'K'                write n words (n = 0: 256) to a, a+1, ...
//   'R' a0 a1                    -> d0 d1 d2 d3         read a word
// Replies queue in a 64-byte FIFO, so the PC may send several commands before reading the
// replies (tools/ttf_host.py keeps at most 16 reads in flight). Writes to the RAMs are
// ignored while the core runs a program (ttf_core).
//
// LEDs (active low): 0 busy, 1 done, 2 heartbeat, 3 UART activity.
// Everything runs on clk (CLK_HZ): the 27 MHz oscillator or the rPLL (boards/tn20k/ttf_gowin.tcl).

`default_nettype none

module ttf_tn20k_top #(
    parameter CLK_HZ = 27000000,
    parameter BAUD   = 115200,
    parameter N      = 8,
    parameter G      = 4,
    parameter PIPE   = 1,
    parameter ACC_W  = 24,
    parameter AW_A   = 11,
    parameter AW_W   = 13,
    parameter W_D    = 6400,
    parameter AW_M   = 4,
    parameter AW_B   = 4,
    parameter AW_D   = 6
) (
    input  wire       clk,
    input  wire       clk_ok,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led_n
);
    localparam DIV = (CLK_HZ + BAUD / 2) / BAUD;

    // ---------------------------------------------------------------- reset
    reg [15:0] por = 16'd0;
    wire       rst = ~por[15];
    always @(posedge clk)
        if (!clk_ok)      por <= 16'd0;
        else if (rst)     por <= por + 1'b1;

    // ---------------------------------------------------------------- UART receive
    reg [2:0]  rx_s = 3'b111;
    always @(posedge clk) rx_s <= {rx_s[1:0], uart_rx};
    wire       rxd = rx_s[2];
    reg [15:0] rx_cnt;
    reg [3:0]  rx_bit;
    reg [7:0]  rx_sh;
    reg        rx_busy, rx_stb;
    always @(posedge clk) begin
        rx_stb <= 1'b0;
        if (rst) begin
            rx_busy <= 1'b0;
        end else if (!rx_busy) begin
            if (!rxd) begin
                rx_busy <= 1'b1; rx_cnt <= DIV / 2; rx_bit <= 4'd0;
            end
        end else if (rx_cnt != 0) begin
            rx_cnt <= rx_cnt - 1'b1;
        end else begin
            rx_cnt <= DIV - 1;
            rx_bit <= rx_bit + 1'b1;
            if (rx_bit == 4'd0) begin
                if (rxd) rx_busy <= 1'b0;                 // not a start bit after all
            end else if (rx_bit <= 4'd8) begin
                rx_sh <= {rxd, rx_sh[7:1]};
            end else begin
                rx_busy <= 1'b0;
                rx_stb  <= rxd;                           // stop bit present
            end
        end
    end

    // ---------------------------------------------------------------- reply FIFO, transmit
    reg  [7:0] fifo [0:63];
    reg  [6:0] f_wp, f_rp;
    wire       f_empty = (f_wp == f_rp);
    reg        f_we;
    reg  [7:0] f_wd;
    always @(posedge clk)
        if (f_we) fifo[f_wp[5:0]] <= f_wd;

    reg [15:0] tx_cnt;
    reg [3:0]  tx_bit;
    reg [9:0]  tx_sh;
    reg        tx_busy;
    assign uart_tx = tx_busy ? tx_sh[0] : 1'b1;
    always @(posedge clk) begin
        if (rst) begin
            f_wp <= 7'd0; f_rp <= 7'd0; tx_busy <= 1'b0;
        end else begin
            if (f_we) f_wp <= f_wp + 1'b1;
            if (!tx_busy) begin
                if (!f_empty) begin
                    tx_sh   <= {1'b1, fifo[f_rp[5:0]], 1'b0};
                    f_rp    <= f_rp + 1'b1;
                    tx_busy <= 1'b1; tx_cnt <= DIV - 1; tx_bit <= 4'd0;
                end
            end else if (tx_cnt != 0) begin
                tx_cnt <= tx_cnt - 1'b1;
            end else begin
                tx_cnt <= DIV - 1;
                tx_sh  <= {1'b1, tx_sh[9:1]};
                tx_bit <= tx_bit + 1'b1;
                if (tx_bit == 4'd9) tx_busy <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------- command decoder
    localparam C_IDLE = 3'd0, C_ADDR = 3'd1, C_CNT = 3'd2, C_DATA = 3'd3, C_RD = 3'd4,
               C_RPLY = 3'd5;
    reg [2:0]  cs;
    reg [7:0]  cmd;
    reg [2:0]  nb;            // bytes of the current field
    reg [15:0] addr;
    reg [31:0] data;
    reg [8:0]  words;         // 'B': words left
    reg        h_we, h_re;
    reg [1:0]  rb;            // reply byte of a read
    reg [31:0] rdat;
    wire [31:0] h_rd;
    wire        h_rv, busy, done;

    always @(posedge clk) begin
        h_we <= 1'b0;
        h_re <= 1'b0;
        f_we <= 1'b0;
        if (rst) begin
            cs <= C_IDLE;
        end else case (cs)
        C_IDLE:
            if (rx_stb) begin
                cmd <= rx_sh; nb <= 3'd0;
                if (rx_sh == "P") begin f_we <= 1'b1; f_wd <= "K"; end
                else if (rx_sh == "W" || rx_sh == "R" || rx_sh == "B") cs <= C_ADDR;
            end
        C_ADDR:
            if (rx_stb) begin
                addr <= {rx_sh, addr[15:8]};
                nb   <= nb + 1'b1;
                if (nb == 3'd1) begin
                    nb <= 3'd0;
                    if (cmd == "R") begin h_re <= 1'b1; cs <= C_RD; end
                    else if (cmd == "B") cs <= C_CNT;
                    else begin words <= 9'd1; cs <= C_DATA; end
                end
            end
        C_CNT:
            if (rx_stb) begin
                words <= (rx_sh == 8'd0) ? 9'd256 : {1'b0, rx_sh};
                cs <= C_DATA;
            end
        C_DATA:
            if (rx_stb) begin
                data <= {rx_sh, data[31:8]};
                nb   <= nb + 1'b1;
                if (nb == 3'd3) begin
                    nb    <= 3'd0;
                    h_we  <= 1'b1;
                    words <= words - 1'b1;
                    if (words == 9'd1) begin
                        f_we <= 1'b1; f_wd <= "K"; cs <= C_IDLE;
                    end
                end
            end
        C_RD:
            if (h_rv) begin
                rdat <= h_rd; rb <= 2'd0; cs <= C_RPLY;
            end
        default: begin // C_RPLY: four bytes into the FIFO
            f_we <= 1'b1;
            f_wd <= rdat[7:0];
            rdat <= {8'd0, rdat[31:8]};
            rb   <= rb + 1'b1;
            if (rb == 2'd3) cs <= C_IDLE;
        end
        endcase
    end

    // h_we fires one clock after the last data byte: the address of a 'B' burst advances
    // after each word
    reg [15:0] w_addr;
    always @(posedge clk) begin
        if (cs == C_ADDR && rx_stb && nb == 3'd1) w_addr <= {rx_sh, addr[15:8]};
        else if (h_we) w_addr <= w_addr + 1'b1;
    end

    ttf_core #(.N(N), .G(G), .PIPE(PIPE), .ACC_W(ACC_W), .AW_A(AW_A), .AW_W(AW_W), .W_D(W_D),
               .AW_M(AW_M), .AW_B(AW_B), .AW_D(AW_D)) u_core (
        .clk(clk), .rst(rst),
        .h_we(h_we), .h_re(h_re), .h_a(h_re ? addr : w_addr), .h_wd(data),
        .h_rd(h_rd), .h_rv(h_rv), .busy(busy), .done(done));

    // ---------------------------------------------------------------- LEDs
    reg [24:0] hb;
    reg [21:0] act;
    always @(posedge clk) begin
        hb <= hb + 1'b1;
        if (rx_stb) act <= 22'h3FFFFF;
        else if (act != 0) act <= act - 1'b1;
    end
    assign led_n = ~{2'b00, act != 0, hb[24], done, busy};
endmodule

`default_nettype wire
