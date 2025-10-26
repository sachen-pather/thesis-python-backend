`timescale 1ns/1ps

// 2-input XOR gate for machine learning training data
module xor_gate(
    input wire a, b,
    output wire y
);
    // Gate primitive implementation
    xor (y, a, b);
endmodule

module testbench;
    reg a, b;
    wire y;
    
    xor_gate dut(
        .a(a),
        .b(b),
        .y(y)
    );
    
    initial begin
        // Test all input combinations
        a = 0; b = 0; #10;
        a = 0; b = 1; #10;
        a = 1; b = 0; #10;
        a = 1; b = 1; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b y=%b", $time, a, b, y);
endmodule