`timescale 1ns/1ps

// Gray code to binary converter (4-bit) for machine learning training data
module gray_to_binary_4bit(
    input wire [3:0] gray,
    output wire [3:0] binary
);
    // Conversion logic
    assign binary[3] = gray[3];
    assign binary[2] = binary[3] ^ gray[2];
    assign binary[1] = binary[2] ^ gray[1];
    assign binary[0] = binary[1] ^ gray[0];
endmodule

module testbench;
    reg [3:0] gray;
    wire [3:0] binary;
    
    gray_to_binary_4bit dut(
        .gray(gray),
        .binary(binary)
    );
    
    initial begin
        // Test all Gray code values 0-15
        gray = 4'b0000; #10;  // Gray 0 -> Binary 0
        gray = 4'b0001; #10;  // Gray 1 -> Binary 1
        gray = 4'b0011; #10;  // Gray 3 -> Binary 2
        gray = 4'b0010; #10;  // Gray 2 -> Binary 3
        gray = 4'b0110; #10;  // Gray 6 -> Binary 4
        gray = 4'b0111; #10;  // Gray 7 -> Binary 5
        gray = 4'b0101; #10;  // Gray 5 -> Binary 6
        gray = 4'b0100; #10;  // Gray 4 -> Binary 7
        gray = 4'b1100; #10;  // Gray 12 -> Binary 8
        gray = 4'b1101; #10;  // Gray 13 -> Binary 9
        gray = 4'b1111; #10;  // Gray 15 -> Binary 10
        gray = 4'b1110; #10;  // Gray 14 -> Binary 11
        $finish;
    end
    
    initial $monitor("Time=%0t gray=%b(%d) binary=%b(%d)", 
                     $time, gray, gray, binary, binary);
endmodule
