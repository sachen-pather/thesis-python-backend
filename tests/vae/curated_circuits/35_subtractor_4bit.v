`timescale 1ns/1ps

// 4-bit magnitude subtractor for machine learning training data
module subtractor_4bit(
    input wire [3:0] a, b,
    input wire bin,
    output wire [3:0] diff,
    output wire bout
);
    wire b1, b2, b3;
    
    // Instantiate 4 full subtractors
    full_subtractor fs0(.a(a[0]), .b(b[0]), .bin(bin), .diff(diff[0]), .bout(b1));
    full_subtractor fs1(.a(a[1]), .b(b[1]), .bin(b1),  .diff(diff[1]), .bout(b2));
    full_subtractor fs2(.a(a[2]), .b(b[2]), .bin(b2),  .diff(diff[2]), .bout(b3));
    full_subtractor fs3(.a(a[3]), .b(b[3]), .bin(b3),  .diff(diff[3]), .bout(bout));
endmodule

// Full subtractor helper module
module full_subtractor(
    input wire a, b, bin,
    output wire diff, bout
);
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
    reg [3:0] a, b;
    reg bin;
    wire [3:0] diff;
    wire bout;
    
    subtractor_4bit dut(
        .a(a),
        .b(b),
        .bin(bin),
        .diff(diff),
        .bout(bout)
    );
    
    initial begin
        // Test various subtractions
        a = 4'b1000; b = 4'b0011; bin = 0; #10;  // 8-3=5
        a = 4'b0110; b = 4'b0010; bin = 0; #10;  // 6-2=4
        a = 4'b1111; b = 4'b0001; bin = 0; #10;  // 15-1=14
        a = 4'b0101; b = 4'b0101; bin = 0; #10;  // 5-5=0
        a = 4'b0011; b = 4'b0111; bin = 0; #10;  // 3-7=-4 (with borrow)
        a = 4'b1010; b = 4'b0011; bin = 1; #10;  // 10-3-1=6
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b(%d) b=%b(%d) bin=%b diff=%b(%d) bout=%b", 
                     $time, a, a, b, b, bin, diff, diff, bout);
endmodule
