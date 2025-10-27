`timescale 1ns/1ps

// 4-bit barrel shifter (left) for machine learning training data
module barrel_shifter_left_4bit(
    input wire [3:0] data_in,
    input wire [1:0] shift_amt,
    output reg [3:0] data_out
);
    // Barrel shifter with variable shift amount
    always @(*) begin
        case(shift_amt)
            2'b00: data_out = data_in;              // No shift
            2'b01: data_out = {data_in[2:0], 1'b0}; // Shift left by 1
            2'b10: data_out = {data_in[1:0], 2'b00}; // Shift left by 2
            2'b11: data_out = {data_in[0], 3'b000};  // Shift left by 3
        endcase
    end
endmodule

module testbench;
    reg [3:0] data_in;
    reg [1:0] shift_amt;
    wire [3:0] data_out;
    
    barrel_shifter_left_4bit dut(
        .data_in(data_in),
        .shift_amt(shift_amt),
        .data_out(data_out)
    );
    
    initial begin
        // Test with different input patterns
        data_in = 4'b1010;
        shift_amt = 2'b00; #10;  // No shift: 1010
        shift_amt = 2'b01; #10;  // Shift 1: 0100
        shift_amt = 2'b10; #10;  // Shift 2: 1000
        shift_amt = 2'b11; #10;  // Shift 3: 0000
        
        data_in = 4'b0011;
        shift_amt = 2'b00; #10;  // No shift: 0011
        shift_amt = 2'b01; #10;  // Shift 1: 0110
        shift_amt = 2'b10; #10;  // Shift 2: 1100
        shift_amt = 2'b11; #10;  // Shift 3: 1000
        
        $finish;
    end
    
    initial $monitor("Time=%0t data_in=%b shift_amt=%b data_out=%b", 
                     $time, data_in, shift_amt, data_out);
endmodule
