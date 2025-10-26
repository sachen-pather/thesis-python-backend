`timescale 1ns/1ps

// D flip-flop for machine learning training data
module d_flip_flop(
    input wire clk,
    input wire d,
    output reg q,
    output wire qbar
);
    // Behavioral implementation
    assign qbar = ~q;
    
    always @(posedge clk) begin
        q <= d;
    end
endmodule

module testbench;
    reg clk, d;
    wire q, qbar;
    
    d_flip_flop dut(
        .clk(clk),
        .d(d),
        .q(q),
        .qbar(qbar)
    );
    
    initial begin
        // Initialize
        clk = 0;
        d = 0;
        #5;
        
        // Test sequence: change d and observe q on clock edges
        d = 1; #10;  // d=1, wait for clock edge
        d = 0; #10;  // d=0, wait for clock edge
        d = 1; #10;  // d=1, wait for clock edge
        d = 1; #10;  // d=1 (no change), wait for clock edge
        d = 0; #10;  // d=0, wait for clock edge
        d = 1; #10;  // d=1, wait for clock edge
        d = 0; #10;  // d=0, wait for clock edge
        
        $finish;
    end
    
    // Clock generation - 10ns period (5ns high, 5ns low)
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b d=%b q=%b qbar=%b", $time, clk, d, q, qbar);
endmodule