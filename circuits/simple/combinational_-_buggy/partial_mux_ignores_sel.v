/*
 * Circuit: Partial Mux (ignores sel)
 * Category: Combinational - Buggy
 * Complexity: SIMPLE
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_mux(input wire a, b, sel, output wire out);
assign out = a;
endmodule
module testbench;
reg a, b, sel; wire out;
bad_mux dut(.a(a), .b(b), .sel(sel), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=0;b=1;sel=0;#10; sel=1;#10; a=1;b=0;sel=0;#10; sel=1;#10; $finish;
end
initial $monitor("Time=%0t a=%b b=%b sel=%b out=%b", $time, a, b, sel, out);
endmodule