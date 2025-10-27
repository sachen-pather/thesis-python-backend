`timescale 1ns/1ps

// 8-to-3 encoder for machine learning training data
module encoder_8_3(
    input wire [7:0] in,
    output reg [2:0] out,
    output reg valid
);
    // Priority encoder behavior
    always @(*) begin
        casex (in)
            8'b00000001: begin out = 3'b000; valid = 1; end
            8'b00000010: begin out = 3'b001; valid = 1; end
            8'b00000100: begin out = 3'b010; valid = 1; end
            8'b00001000: begin out = 3'b011; valid = 1; end
            8'b00010000: begin out = 3'b100; valid = 1; end
            8'b00100000: begin out = 3'b101; valid = 1; end
            8'b01000000: begin out = 3'b110; valid = 1; end
            8'b10000000: begin out = 3'b111; valid = 1; end
            default:     begin out = 3'b000; valid = 0; end
        endcase
    end
endmodule

module testbench;
    reg [7:0] in;
    wire [2:0] out;
    wire valid;
    
    encoder_8_3 dut(
        .in(in),
        .out(out),
        .valid(valid)
    );
    
    initial begin
        // Test all valid one-hot inputs
        in = 8'b00000000; #10;  // Invalid
        in = 8'b00000001; #10;  // 0
        in = 8'b00000010; #10;  // 1
        in = 8'b00000100; #10;  // 2
        in = 8'b00001000; #10;  // 3
        in = 8'b00010000; #10;  // 4
        in = 8'b00100000; #10;  // 5
        in = 8'b01000000; #10;  // 6
        in = 8'b10000000; #10;  // 7
        in = 8'b00000011; #10;  // Invalid (multiple bits)
        $finish;
    end
    
    initial $monitor("Time=%0t in=%b out=%b valid=%b", $time, in, out, valid);
endmodule
