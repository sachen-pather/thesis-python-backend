`timescale 1ns/1ps

// 2:4 decoder for machine learning training data
module decoder_2_4(
    input wire [1:0] in,
    input wire en,
    output reg [3:0] out
);
    // Behavioral implementation with case statement
    always @(*) begin
        if (en) begin
            case (in)
                2'b00: out = 4'b0001;
                2'b01: out = 4'b0010;
                2'b10: out = 4'b0100;
                2'b11: out = 4'b1000;
            endcase
        end else begin
            out = 4'b0000;
        end
    end
endmodule

module testbench;
    reg [1:0] in;
    reg en;
    wire [3:0] out;
    
    decoder_2_4 dut(
        .in(in),
        .en(en),
        .out(out)
    );
    
    initial begin
        // Test with enable off and on
        en = 0; in = 2'b00; #10;
        en = 1; in = 2'b00; #10;
        in = 2'b01; #10;
        in = 2'b10; #10;
        in = 2'b11; #10;
        en = 0; in = 2'b10; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t en=%b in=%b out=%b", $time, en, in, out);
endmodule
