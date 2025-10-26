/*
 * Circuit: Simple ALU with Flags
 * Category: CPU Components - Normal
 * Complexity: COMPLEX
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module alu_with_flags(input wire [7:0] a, b, input wire [2:0] op, 
                      output reg [7:0] result, output reg zero, carry, negative);
always @(*) begin
    case (op)
        3'b000: {carry, result} = a + b;
        3'b001: {carry, result} = a - b;
        3'b010: result = a & b;
        3'b011: result = a | b;
        3'b100: result = a ^ b;
        3'b101: result = ~a;
        3'b110: result = a << 1;
        3'b111: result = a >> 1;
        default: result = 0;
    endcase
    zero = (result == 0);
    negative = result[7];
    if (op > 3'b001) carry = 0;
end
endmodule

module testbench;
reg [7:0] a, b; reg [2:0] op; wire [7:0] result; wire zero, carry, negative;
alu_with_flags dut(.a(a), .b(b), .op(op), .result(result), .zero(zero), .carry(carry), .negative(negative));
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    a=8'h0F; b=8'h01; op=3'b000; #10; // ADD
    a=8'h10; b=8'h10; op=3'b001; #10; // SUB (should set zero)
    a=8'hFF; b=8'h01; op=3'b000; #10; // ADD with carry
    a=8'hAA; b=8'h55; op=3'b010; #10; // AND
    a=8'h80; b=8'h00; op=3'b111; #10; // SHR (should set negative)
    $finish;
end
initial $monitor("Time=%0t op=%b a=%h b=%h result=%h Z=%b C=%b N=%b", 
                 $time, op, a, b, result, zero, carry, negative);
endmodule