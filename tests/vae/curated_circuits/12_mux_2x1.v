`timescale 1ns/1ps

// 2:1 multiplexer for machine learning training data
module mux_2x1(
    input wire a, b,
    input wire sel,
    output reg y
);
    // Behavioral implementation
    always @(*) begin
        case(sel)
            1'b0: y = a;
            1'b1: y = b;
        endcase
    end
endmodule

module testbench;
    reg a, b, sel;
    wire y;
    
    mux_2x1 dut(
        .a(a),
        .b(b),
        .sel(sel),
        .y(y)
    );
    
    initial begin
        // Test with different input combinations
        a = 0; b = 1; sel = 0; #10;  // Select a
        sel = 1; #10;                 // Select b
        a = 1; b = 0; sel = 0; #10;  // Select a
        sel = 1; #10;                 // Select b
        a = 1; b = 1; sel = 0; #10;  // Select a
        sel = 1; #10;                 // Select b
        a = 0; b = 0; sel = 0; #10;  // Select a
        sel = 1; #10;                 // Select b
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b sel=%b y=%b", $time, a, b, sel, y);
endmodule
