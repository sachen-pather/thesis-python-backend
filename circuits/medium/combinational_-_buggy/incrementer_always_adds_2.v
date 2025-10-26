/*
 * Circuit: Incrementer (always adds 2)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_incrementer(input wire [3:0] a, output wire [3:0] out, output wire overflow);
assign {overflow, out} = a + 4'b0010;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire overflow;
bad_incrementer dut(.a(a), .out(out), .overflow(overflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd0;#10; a=4'd7;#10; a=4'd14;#10; a=4'd15;#10; a=4'd3;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d overflow=%b", $time, a, out, overflow);
endmodule