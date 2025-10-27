`timescale 1ns/1ps

// Half adder for machine learning training data
module half_adder(
    input wire a, b,
    output wire sum, carry
);
    // Gate-level implementation
    xor (sum, a, b);
    and (carry, a, b);
endmodule

module testbench;
    reg a, b;
    wire sum, carry;
    
    half_adder dut(
        .a(a),
        .b(b),
        .sum(sum),
        .carry(carry)
    );
    
    initial begin
        // Test all input combinations
        a = 0; b = 0; #10;
        a = 0; b = 1; #10;
        a = 1; b = 0; #10;
        a = 1; b = 1; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b sum=%b carry=%b", $time, a, b, sum, carry);
endmodule
