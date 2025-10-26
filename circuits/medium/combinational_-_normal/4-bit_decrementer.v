/*
 * Circuit: 4-bit Decrementer
 * Category: Combinational - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module decrementer(input wire [3:0] a, output wire [3:0] out, output wire underflow);
assign {underflow, out} = {1'b0, a} - 5'b00001;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire underflow;
decrementer dut(.a(a), .out(out), .underflow(underflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5;#10; a=4'd1;#10; a=4'd0;#10; a=4'd15;#10; a=4'd8;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d underflow=%b", $time, a, out, underflow);
endmodule