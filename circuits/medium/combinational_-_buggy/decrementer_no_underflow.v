/*
 * Circuit: Decrementer (no underflow)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_decrementer(input wire [3:0] a, output wire [3:0] out, output wire underflow);
assign out = a - 4'b0001;
assign underflow = 1'b0;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire underflow;
bad_decrementer dut(.a(a), .out(out), .underflow(underflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5;#10; a=4'd1;#10; a=4'd0;#10; a=4'd15;#10; a=4'd8;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d underflow=%b", $time, a, out, underflow);
endmodule