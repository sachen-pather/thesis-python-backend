`timescale 1ns/1ps

// 3:8 decoder for machine learning training data
module decoder_3_8(
    input wire [2:0] in,
    output reg [7:0] out
);
    // Behavioral implementation with case statement
    always @(*) begin
        case (in)
            3'b000: out = 8'b00000001;
            3'b001: out = 8'b00000010;
            3'b010: out = 8'b00000100;
            3'b011: out = 8'b00001000;
            3'b100: out = 8'b00010000;
            3'b101: out = 8'b00100000;
            3'b110: out = 8'b01000000;
            3'b111: out = 8'b10000000;
        endcase
    end
endmodule

module testbench;
    reg [2:0] in;
    wire [7:0] out;
    
    decoder_3_8 dut(
        .in(in),
        .out(out)
    );
    
    initial begin
        // Test all 8 input combinations
        in = 3'b000; #10;
        in = 3'b001; #10;
        in = 3'b010; #10;
        in = 3'b011; #10;
        in = 3'b100; #10;
        in = 3'b101; #10;
        in = 3'b110; #10;
        in = 3'b111; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t in=%b out=%b", $time, in, out);
endmodule