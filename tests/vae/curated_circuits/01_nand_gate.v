`timescale 1ns/1ps

// 2-input NAND gate for machine learning training data
module nand_gate(
    input wire a, b,
    output wire y
);
    // Gate primitive implementation
    nand (y, a, b);
endmodule

module testbench;
    reg a, b;
    wire y;
    
    nand_gate dut(
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
