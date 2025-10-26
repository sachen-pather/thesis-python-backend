/*
 * Circuit: 4-bit Barrel Shifter
 * Category: Combinational - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module barrel_shifter(input wire [3:0] in, input wire [1:0] shift, input wire dir, output reg [3:0] out);
always @(*) begin
    if (dir) // Right shift
        out = in >> shift;
    else     // Left shift
        out = in << shift;
end
endmodule
module testbench;
reg [3:0] in; reg [1:0] shift; reg dir; wire [3:0] out;
barrel_shifter dut(.in(in), .shift(shift), .dir(dir), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b1101; dir=0; shift=2'd0;#10; shift=2'd1;#10; shift=2'd2;#10;
    dir=1; shift=2'd0;#10; shift=2'd1;#10; shift=2'd2;#10; $finish;
end
initial $monitor("Time=%0t in=%b dir=%b shift=%d out=%b", $time, in, dir, shift, out);
endmodule