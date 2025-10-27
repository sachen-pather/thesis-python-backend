`timescale 1ns/1ps

// 8:1 multiplexer for machine learning training data
module mux_8x1(
    input wire [7:0] in,
    input wire [2:0] sel,
    output reg out
);
    // Behavioral implementation with case statement
    always @(*) begin
        case(sel)
            3'b000: out = in[0];
            3'b001: out = in[1];
            3'b010: out = in[2];
            3'b011: out = in[3];
            3'b100: out = in[4];
            3'b101: out = in[5];
            3'b110: out = in[6];
            3'b111: out = in[7];
        endcase
    end
endmodule

module testbench;
    reg [7:0] in;
    reg [2:0] sel;
    wire out;
    
    mux_8x1 dut(
        .in(in),
        .sel(sel),
        .out(out)
    );
    
    initial begin
        // Set input pattern and test all select combinations
        in = 8'b10110100;
        sel = 3'b000; #10;
        sel = 3'b001; #10;
        sel = 3'b010; #10;
        sel = 3'b011; #10;
        sel = 3'b100; #10;
        sel = 3'b101; #10;
        sel = 3'b110; #10;
        sel = 3'b111; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t in=%b sel=%b out=%b", $time, in, sel, out);
endmodule
