`timescale 1ns/1ps

// 4:2 priority encoder for machine learning training data
module priority_encoder_4_2(
    input wire [3:0] in,
    output reg [1:0] out,
    output reg valid
);
    // Behavioral implementation with casex
    always @(*) begin
        casex (in)
            4'b1xxx: begin out = 2'b11; valid = 1; end
            4'b01xx: begin out = 2'b10; valid = 1; end
            4'b001x: begin out = 2'b01; valid = 1; end
            4'b0001: begin out = 2'b00; valid = 1; end
            default: begin out = 2'b00; valid = 0; end
        endcase
    end
endmodule

module testbench;
    reg [3:0] in;
    wire [1:0] out;
    wire valid;
    
    priority_encoder_4_2 dut(
        .in(in),
        .out(out),
        .valid(valid)
    );
    
    initial begin
        // Test various input combinations
        in = 4'b0000; #10;
        in = 4'b0001; #10;
        in = 4'b0010; #10;
        in = 4'b0100; #10;
        in = 4'b1000; #10;
        in = 4'b0011; #10;
        in = 4'b1010; #10;
        in = 4'b1111; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t in=%b out=%b valid=%b", $time, in, out, valid);
endmodule
