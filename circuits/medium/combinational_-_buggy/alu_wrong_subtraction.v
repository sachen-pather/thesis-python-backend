/*
 * Circuit: ALU (wrong subtraction)
 * Category: Combinational - Buggy
 * Complexity: MEDIUM
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_alu(input wire [3:0] a, b, input wire [1:0] op, output reg [3:0] out);
always @(*) begin
    case(op)
        2'b00: out = a + b;
        2'b01: out = b - a;
        2'b10: out = a & b;
        2'b11: out = a | b;
    endcase
end
endmodule
module testbench;
reg [3:0] a, b; reg [1:0] op; wire [3:0] out;
bad_alu dut(.a(a), .b(b), .op(op), .out(out));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=4'd5; b=4'd3; op=2'd0;#10; op=2'd1;#10; op=2'd2;#10; op=2'd3;#10;
    a=4'd12; b=4'd7; op=2'd0;#10; op=2'd1;#10; $finish;
end
initial $monitor("Time=%0t a=%d b=%d op=%d out=%d", $time, a, b, op, out);
endmodule