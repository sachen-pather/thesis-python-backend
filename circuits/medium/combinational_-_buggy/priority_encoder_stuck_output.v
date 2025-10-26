/*
 * Circuit: Priority Encoder (stuck output)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_priority_encoder(input wire [3:0] in, output reg [1:0] out, output reg valid);
always @(*) begin
    out = 2'd2;
    valid = (in != 4'b0000);
end
endmodule
module testbench;
reg [3:0] in; wire [1:0] out; wire valid;
bad_priority_encoder dut(.in(in), .out(out), .valid(valid));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    in=4'b0000;#10; in=4'b0001;#10; in=4'b0010;#10; in=4'b0100;#10;
    in=4'b1000;#10; in=4'b1111;#10; in=4'b0101;#10; $finish;
end
initial $monitor("Time=%0t in=%b out=%d valid=%b", $time, in, out, valid);
endmodule