`timescale 1ns/1ps

// Parity checker for machine learning training data
module parity_checker(
    input wire [7:0] data,
    input wire parity_bit,
    output wire error
);
    // Check if total number of 1s (including parity) is odd
    // Error is 1 if parity check fails
    assign error = ^{data, parity_bit};
endmodule

module testbench;
    reg [7:0] data;
    reg parity_bit;
    wire error;
    
    parity_checker dut(
        .data(data),
        .parity_bit(parity_bit),
        .error(error)
    );
    
    initial begin
        // Test correct even parity (no error)
        data = 8'b00000000; parity_bit = 0; #10;  // 0 ones + 0 = even (no error)
        data = 8'b00000001; parity_bit = 1; #10;  // 1 one  + 1 = even (no error)
        data = 8'b00000011; parity_bit = 0; #10;  // 2 ones + 0 = even (no error)
        
        // Test incorrect parity (error)
        data = 8'b00000000; parity_bit = 1; #10;  // 0 ones + 1 = odd (error!)
        data = 8'b00000001; parity_bit = 0; #10;  // 1 one  + 0 = odd (error!)
        
        // More correct cases
        data = 8'b11111111; parity_bit = 0; #10;  // 8 ones + 0 = even (no error)
        data = 8'b10101010; parity_bit = 0; #10;  // 4 ones + 0 = even (no error)
        data = 8'b01010101; parity_bit = 0; #10;  // 4 ones + 0 = even (no error)
        
        $finish;
    end
    
    initial $monitor("Time=%0t data=%b parity_bit=%b error=%b", 
                     $time, data, parity_bit, error);
endmodule
