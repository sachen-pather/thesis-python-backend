`timescale 1ns/1ps

// JK flip-flop for machine learning training data
module jk_flip_flop(
    input wire clk,
    input wire j, k,
    output reg q,
    output wire qbar
);
    // Behavioral implementation
    assign qbar = ~q;
    
    always @(posedge clk) begin
        case({j,k})
            2'b00: q <= q;      // Hold
            2'b01: q <= 1'b0;   // Reset
            2'b10: q <= 1'b1;   // Set
            2'b11: q <= ~q;     // Toggle
        endcase
    end
endmodule

module testbench;
    reg clk, j, k;
    wire q, qbar;
    
    jk_flip_flop dut(
        .clk(clk),
        .j(j),
        .k(k),
        .q(q),
        .qbar(qbar)
    );
    
    initial begin
        // Initialize
        clk = 0;
        j = 0; k = 0;
        #5;
        
        // Test all JK combinations
        j = 0; k = 0; #10;  // Hold state
        j = 0; k = 1; #10;  // Reset (q=0)
        j = 1; k = 0; #10;  // Set (q=1)
        j = 1; k = 1; #10;  // Toggle
        j = 1; k = 1; #10;  // Toggle again
        j = 0; k = 0; #10;  // Hold
        j = 0; k = 1; #10;  // Reset again
        j = 1; k = 1; #10;  // Toggle from 0
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b j=%b k=%b q=%b qbar=%b", $time, clk, j, k, q, qbar);
endmodule