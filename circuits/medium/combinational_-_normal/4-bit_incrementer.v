/*
 * Circuit: 4-bit Incrementer
 * Category: Combinational - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module incrementer(input wire [3:0] a, output wire [3:0] out, output wire overflow);
assign {overflow, out} = a + 4'b0001;
endmodule
module testbench;
reg [3:0] a; wire [3:0] out; wire overflow;
incrementer dut(.a(a), .out(out), .overflow(overflow));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd0;#10; a=4'd7;#10; a=4'd14;#10; a=4'd15;#10; a=4'd3;#10; $finish;
end
initial $monitor("Time=%0t a=%d out=%d overflow=%b", $time, a, out, overflow);
endmodule