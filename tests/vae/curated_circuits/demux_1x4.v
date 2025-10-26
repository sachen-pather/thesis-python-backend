`timescale 1ns/1ps

// 1:4 demultiplexer for machine learning training data
module demux_1x4(
    input wire y,
    input wire [1:0] sel,
    output reg a, b, c, d
);
    // Behavioral implementation with case statement
    always @(*) begin
        case(sel)
            2'b00: begin
                a = y; b = 0; c = 0; d = 0;
            end
            2'b01: begin
                a = 0; b = y; c = 0; d = 0;
            end
            2'b10: begin
                a = 0; b = 0; c = y; d = 0;
            end
            2'b11: begin
                a = 0; b = 0; c = 0; d = y;
            end
        endcase
    end
endmodule

module testbench;
    reg y;
    reg [1:0] sel;
    wire a, b, c, d;
    
    demux_1x4 dut(
        .y(y),
        .sel(sel),
        .a(a),
        .b(b),
        .c(c),
        .d(d)
    );
    
    initial begin
        // Test with input y=0
        y = 0;
        sel = 2'b00; #10;  // Route to a
        sel = 2'b01; #10;  // Route to b
        sel = 2'b10; #10;  // Route to c
        sel = 2'b11; #10;  // Route to d
        
        // Test with input y=1
        y = 1;
        sel = 2'b00; #10;  // Route to a
        sel = 2'b01; #10;  // Route to b
        sel = 2'b10; #10;  // Route to c
        sel = 2'b11; #10;  // Route to d
        
        $finish;
    end
    
    initial $monitor("Time=%0t y=%b sel=%b a=%b b=%b c=%b d=%b", $time, y, sel, a, b, c, d);
endmodule