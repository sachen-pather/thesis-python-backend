`timescale 1ns/1ps

// 4:1 multiplexer for machine learning training data
module mux_4x1(
    input wire a, b, c, d,
    input wire [1:0] sel,
    output reg y
);
    // Behavioral implementation with case statement
    always @(*) begin
        case(sel)
            2'b00: y = a;
            2'b01: y = b;
            2'b10: y = c;
            2'b11: y = d;
        endcase
    end
endmodule

module testbench;
    reg a, b, c, d;
    reg [1:0] sel;
    wire y;
    
    mux_4x1 dut(
        .a(a),
        .b(b),
        .c(c),
        .d(d),
        .sel(sel),
        .y(y)
    );
    
    initial begin
        // Set different values on inputs
        a = 0; b = 1; c = 0; d = 1;
        
        // Test all select combinations
        sel = 2'b00; #10;  // Select a
        sel = 2'b01; #10;  // Select b
        sel = 2'b10; #10;  // Select c
        sel = 2'b11; #10;  // Select d
        
        // Change inputs and test again
        a = 1; b = 0; c = 1; d = 0; #10;
        sel = 2'b00; #10;  // Select a
        sel = 2'b01; #10;  // Select b
        sel = 2'b10; #10;  // Select c
        sel = 2'b11; #10;  // Select d
        
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b b=%b c=%b d=%b sel=%b y=%b", $time, a, b, c, d, sel, y);
endmodule