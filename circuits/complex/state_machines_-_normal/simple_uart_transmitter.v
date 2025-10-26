/*
 * Circuit: Simple UART Transmitter
 * Category: State Machines - Normal
 * Complexity: COMPLEX
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module uart_tx(input wire clk, rst, start, input wire [7:0] data, output reg tx, busy);
localparam IDLE=0, START=1, DATA=2, STOP=3;
reg [1:0] state;
reg [2:0] bit_idx;
reg [7:0] shift_reg;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        tx <= 1;
        busy <= 0;
        bit_idx <= 0;
    end else begin
        case (state)
            IDLE: begin
                tx <= 1;
                if (start) begin
                    state <= START;
                    busy <= 1;
                    shift_reg <= data;
                end
            end
            START: begin
                tx <= 0;
                state <= DATA;
                bit_idx <= 0;
            end
            DATA: begin
                tx <= shift_reg[bit_idx];
                if (bit_idx == 7) state <= STOP;
                else bit_idx <= bit_idx + 1;
            end
            STOP: begin
                tx <= 1;
                state <= IDLE;
                busy <= 0;
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst, start; reg [7:0] data; wire tx, busy;
uart_tx dut(.clk(clk), .rst(rst), .start(start), .data(data), .tx(tx), .busy(busy));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; start=0; data=8'h00; #10; rst=0;
    #10; data=8'hA5; start=1; #10; start=0;
    wait(!busy); #50;
    data=8'h3C; start=1; #10; start=0;
    wait(!busy); #50; $finish;
end
initial $monitor("Time=%0t start=%b data=%h tx=%b busy=%b state=%d", $time, start, data, tx, busy, dut.state);
endmodule