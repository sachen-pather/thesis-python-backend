`timescale 1ns/1ps

// Full adder for machine learning training data
module full_adder(
    input wire a,
    input wire b, 
    input wire cin,
    output wire sum,
    output wire cout
);
    // Gate-level implementation
    wire w1, w2, w3;
    
    xor xor1(w1, a, b);
    xor xor2(sum, w1, cin);
    and and1(w2, w1, cin);
    and and2(w3, a, b);
    or  or1(cout, w2, w3);
endmodule

module testbench;
    reg a, b, cin;
    wire sum, cout;
    
    full_adder dut(
        .a(a),
        .b(b),
        .cin(cin),
        .sum(sum),
        .cout(cout)
    );
    
    initial begin
        // Test all 8 possible input combinations
        a = 0; b = 0; cin = 0; #10;  // 0+0+0 = 00
        a = 0; b = 0; cin = 1; #10;  // 0+0+1 = 01
        a = 0; b = 1; cin = 0; #10;  // 0+1+0 = 01
        a = 0; b = 1; cin = 1; #10;  // 0+1+1 = 10
        a = 1; b = 0; cin = 0; #10;  // 1+0+0 = 01
        a = 1; b = 0; cin = 1; #10;  // 1+0+1 = 10
        a = 1; b = 1; cin = 0; #10;  // 1+1+0 = 10
        a = 1; b = 1; cin = 1; #10;  // 1+1+1 = 11
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b cin=%b sum=%b cout=%b result=%d", 
                     $time, a, b, cin, sum, cout, {cout,sum});
endmodule