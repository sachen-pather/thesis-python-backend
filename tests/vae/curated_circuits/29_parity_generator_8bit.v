`timescale 1ns/1ps

// 8-bit parity generator for machine learning training data
module parity_generator_8bit(
    input wire [7:0] data,
    output wire even_parity,
    output wire odd_parity
);
    // Even parity: XOR of all bits
    assign even_parity = ^data;
    
    // Odd parity: complement of even parity
    assign odd_parity = ~even_parity;
endmodule

module testbench;
    reg [7:0] data;
    wire even_parity, odd_parity;
    
    parity_generator_8bit dut(
        .data(data),
        .even_parity(even_parity),
        .odd_parity(odd_parity)
    );
    
    initial begin
        // Test various patterns
        data = 8'b00000000; #10;  // 0 ones -> even=0, odd=1
        data = 8'b00000001; #10;  // 1 one  -> even=1, odd=0
        data = 8'b00000011; #10;  // 2 ones -> even=0, odd=1
        data = 8'b10101010; #10;  // 4 ones -> even=0, odd=1
        data = 8'b11111111; #10;  // 8 ones -> even=0, odd=1
        data = 8'b10000001; #10;  // 2 ones -> even=0, odd=1
        data = 8'b11100111; #10;  // 6 ones -> even=0, odd=1
        data = 8'b01010101; #10;  // 4 ones -> even=0, odd=1
        $finish;
    end
    
    initial $monitor("Time=%0t data=%b even_parity=%b odd_parity=%b", 
                     $time, data, even_parity, odd_parity);
endmodule
