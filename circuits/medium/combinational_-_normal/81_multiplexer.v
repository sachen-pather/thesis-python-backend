/*
 * Circuit: 8:1 Multiplexer
 * Category: Combinational - Normal
 * Complexity: MEDIUM
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module mux8to1(input wire [7:0] in, input wire [2:0] sel, output wire out);
assign out = in[sel];
endmodule
module testbench;
reg [7:0] in; reg [2:0] sel; wire out;
mux8to1 dut(.in(in), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=8'b10101100; sel=3'd0;#10; sel=3'd1;#10; sel=3'd2;#10; sel=3'd3;#10;
    sel=3'd4;#10; sel=3'd5;#10; sel=3'd6;#10; sel=3'd7;#10; $finish;
end
initial $monitor("Time=%0t in=%b sel=%d out=%b", $time, in, sel, out);
endmodule