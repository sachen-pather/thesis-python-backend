`timescale 1ns/1ps

// Simple 4-bit ALU for machine learning training data
module alu_4bit(
    input wire [3:0] a, b,
    input wire [2:0] op,
    output reg [3:0] result,
    output reg zero
);
    // ALU operations
    parameter ADD  = 3'b000;
    parameter SUB  = 3'b001;
    parameter AND  = 3'b010;
    parameter OR   = 3'b011;
    parameter XOR  = 3'b100;
    parameter SHL  = 3'b101;  // Shift left
    parameter SHR  = 3'b110;  // Shift right
    parameter NOT  = 3'b111;  // NOT A
    
    always @(*) begin
        case(op)
            ADD:  result = a + b;
            SUB:  result = a - b;
            AND:  result = a & b;
            OR:   result = a | b;
            XOR:  result = a ^ b;
            SHL:  result = a << 1;
            SHR:  result = a >> 1;
            NOT:  result = ~a;
        endcase
        
        zero = (result == 4'b0000);
    end
endmodule

module testbench;
    reg [3:0] a, b;
    reg [2:0] op;
    wire [3:0] result;
    wire zero;
    
    alu_4bit dut(
        .a(a),
        .b(b),
        .op(op),
        .result(result),
        .zero(zero)
    );
    
    initial begin
        // Test all operations
        a = 4'b0101; b = 4'b0011;
        
        op = 3'b000; #10;  // ADD: 5+3=8
        op = 3'b001; #10;  // SUB: 5-3=2
        op = 3'b010; #10;  // AND: 0101 & 0011 = 0001
        op = 3'b011; #10;  // OR:  0101 | 0011 = 0111
        op = 3'b100; #10;  // XOR: 0101 ^ 0011 = 0110
        op = 3'b101; #10;  // SHL: 0101 << 1 = 1010
        op = 3'b110; #10;  // SHR: 0101 >> 1 = 0010
        op = 3'b111; #10;  // NOT: ~0101 = 1010
        
        // Test zero flag
        a = 4'b0011; b = 4'b0011;
        op = 3'b001; #10;  // SUB: 3-3=0 (zero=1)
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b op=%b result=%b zero=%b", 
                     $time, a, b, op, result, zero);
endmodule
