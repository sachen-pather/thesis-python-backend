`timescale 1ns/1ps

// Full adder module (needed for ripple carry adder)
module full_adder(
    input wire a, b, cin,
    output wire sum, cout
);
    wire w1, w2, w3;
    
    xor xor1(w1, a, b);
    xor xor2(sum, w1, cin);
    and and1(w2, w1, cin);
    and and2(w3, a, b);
    or  or1(cout, w2, w3);
endmodule

// 4-bit ripple carry adder for machine learning training data
module ripple_carry_adder_4bit(
    input wire [3:0] a, b,
    input wire cin,
    output wire [3:0] sum,
    output wire cout
);
    wire c1, c2, c3;
    
    full_adder fa0(.a(a[0]), .b(b[0]), .cin(cin), .sum(sum[0]), .cout(c1));
    full_adder fa1(.a(a[1]), .b(b[1]), .cin(c1), .sum(sum[1]), .cout(c2));
    full_adder fa2(.a(a[2]), .b(b[2]), .cin(c2), .sum(sum[2]), .cout(c3));
    full_adder fa3(.a(a[3]), .b(b[3]), .cin(c3), .sum(sum[3]), .cout(cout));
endmodule

module testbench;
    reg [3:0] a, b;
    reg cin;
    wire [3:0] sum;
    wire cout;
    
    ripple_carry_adder_4bit dut(
        .a(a),
        .b(b),
        .cin(cin),
        .sum(sum),
        .cout(cout)
    );
    
    initial begin
        // Test various addition cases
        a = 4'b0011; b = 4'b0101; cin = 0; #10;  // 3 + 5 = 8
        a = 4'b1111; b = 4'b0001; cin = 0; #10;  // 15 + 1 = 16
        a = 4'b1010; b = 4'b0110; cin = 1; #10;  // 10 + 6 + 1 = 17
        a = 4'b0111; b = 4'b1001; cin = 0; #10;  // 7 + 9 = 16
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b cin=%b sum=%b cout=%b result=%d", 
                     $time, a, b, cin, sum, cout, {cout,sum});
endmodule
