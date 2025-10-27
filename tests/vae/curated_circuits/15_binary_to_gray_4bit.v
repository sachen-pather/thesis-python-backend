`timescale 1ns/1ps

// Binary to Gray code converter (4-bit) for machine learning training data
module binary_to_gray_4bit(
    input wire [3:0] binary,
    output wire [3:0] gray
);
    // Conversion logic
    assign gray[3] = binary[3];
    assign gray[2] = binary[3] ^ binary[2];
    assign gray[1] = binary[2] ^ binary[1];
    assign gray[0] = binary[1] ^ binary[0];
endmodule

module testbench;
    reg [3:0] binary;
    wire [3:0] gray;
    
    binary_to_gray_4bit dut(
        .binary(binary),
        .gray(gray)
    );
    
    initial begin
        // Test all binary values 0-15
        binary = 4'b0000; #10;  // Binary 0 -> Gray 0
        binary = 4'b0001; #10;  // Binary 1 -> Gray 1
        binary = 4'b0010; #10;  // Binary 2 -> Gray 3
        binary = 4'b0011; #10;  // Binary 3 -> Gray 2
        binary = 4'b0100; #10;  // Binary 4 -> Gray 6
        binary = 4'b0101; #10;  // Binary 5 -> Gray 7
        binary = 4'b0110; #10;  // Binary 6 -> Gray 5
        binary = 4'b0111; #10;  // Binary 7 -> Gray 4
        binary = 4'b1000; #10;  // Binary 8 -> Gray 12
        binary = 4'b1001; #10;  // Binary 9 -> Gray 13
        binary = 4'b1010; #10;  // Binary 10 -> Gray 15
        binary = 4'b1011; #10;  // Binary 11 -> Gray 14
        $finish;
    end
    
    initial $monitor("Time=%0t binary=%b(%d) gray=%b(%d)", 
                     $time, binary, binary, gray, gray);
endmodule
