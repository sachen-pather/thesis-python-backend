/*
 * Circuit: 4:1 Mux (wrong bit order)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_mux4to1(input wire [3:0] in, input wire [1:0] sel, output wire out);
assign out = in[3-sel];
endmodule
module testbench;
reg [3:0] in; reg [1:0] sel; wire out;
bad_mux4to1 dut(.in(in), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b1010; sel=2'd0;#10; sel=2'd1;#10; sel=2'd2;#10; sel=2'd3;#10;
    in=4'b0110; sel=2'd0;#10; sel=2'd2;#10; $finish;
end
initial $monitor("Time=%0t in=%b sel=%d out=%b", $time, in, sel, out);
endmodule