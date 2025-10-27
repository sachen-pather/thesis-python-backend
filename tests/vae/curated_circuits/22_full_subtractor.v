`timescale 1ns/1ps

// Full subtractor for machine learning training data
module full_subtractor(
    input wire a, b, bin,
    output wire diff, bout
);
    // Gate-level implementation
    wire w1, w2, w3, w4, w5;
    
    xor xor1(w1, a, b);
    xor xor2(diff, w1, bin);
    
    not not1(w2, a);
    and and1(w3, w2, b);
    not not2(w4, w1);
    and and2(w5, w4, bin);
    or  or1(bout, w3, w5);
endmodule

module testbench;
    reg a, b, bin;
    wire diff, bout;
    
    full_subtractor dut(
        .a(a),
        .b(b),
        .bin(bin),
        .diff(diff),
        .bout(bout)
    );
    
    initial begin
        // Test all 8 possible input combinations
        a = 0; b = 0; bin = 0; #10;  // 0-0-0 = 0
        a = 0; b = 0; bin = 1; #10;  // 0-0-1 = -1 (diff=1, bout=1)
        a = 0; b = 1; bin = 0; #10;  // 0-1-0 = -1
        a = 0; b = 1; bin = 1; #10;  // 0-1-1 = -2
        a = 1; b = 0; bin = 0; #10;  // 1-0-0 = 1
        a = 1; b = 0; bin = 1; #10;  // 1-0-1 = 0
        a = 1; b = 1; bin = 0; #10;  // 1-1-0 = 0
        a = 1; b = 1; bin = 1; #10;  // 1-1-1 = -1
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b bin=%b diff=%b bout=%b", 
                     $time, a, b, bin, diff, bout);
endmodule
