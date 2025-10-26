/*
 * Circuit: Comparator (inverted gt-lt)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_comparator(input wire [3:0] a, b, output wire eq, gt, lt);
assign eq = (a == b);
assign gt = (a < b);
assign lt = (a > b);
endmodule
module testbench;
reg [3:0] a, b; wire eq, gt, lt;
bad_comparator dut(.a(a), .b(b), .eq(eq), .gt(gt), .lt(lt));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd5;#10; a=4'd8; b=4'd3;#10; a=4'd2; b=4'd7;#10;
    a=4'd15;b=4'd0;#10; a=4'd0; b=4'd15;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d eq=%b gt=%b lt=%b", $time, a, b, eq, gt, lt);
endmodule