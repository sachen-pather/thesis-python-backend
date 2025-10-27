`timescale 1ns/1ps

// NOT gate for machine learning training data
module not_gate(
    input wire a,
    output wire y
);
    // Gate primitive implementation
    not (y, a);
endmodule

module testbench;
    reg a;
    wire y;
    
    not_gate dut(
        .a(a),
        .y(y)
    );
    
    initial begin
        // Test both input combinations
        a = 0; #10;
        a = 1; #10;
        a = 0; #10;
        a = 1; #10;
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b y=%b", $time, a, y);
endmodule
