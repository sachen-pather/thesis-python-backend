`timescale 1ns/1ps

// T flip-flop for machine learning training data
module t_flip_flop(
    input wire clk,
    input wire t,
    output reg q,
    output wire qbar
);
    // Behavioral implementation
    assign qbar = ~q;
    
    always @(posedge clk) begin
        case(t)
            1'b0: q <= q;      // Hold
            1'b1: q <= ~q;     // Toggle
        endcase
    end
endmodule

module testbench;
    reg clk, t;
    wire q, qbar;
    
    t_flip_flop dut(
        .clk(clk),
        .t(t),
        .q(q),
        .qbar(qbar)
    );
    
    initial begin
        // Initialize
        clk = 0;
        t = 0;
        #5;
        
        // Force initial state for q (since flip-flops start undefined)
        force dut.q = 0;
        #1;
        release dut.q;
        
        // Test T flip-flop operations
        t = 0; #10;  // Hold state
        t = 1; #10;  // Toggle
        t = 1; #10;  // Toggle again
        t = 0; #10;  // Hold
        t = 1; #10;  // Toggle
        t = 0; #10;  // Hold
        t = 1; #10;  // Toggle
        t = 1; #10;  // Toggle again
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clk=%b t=%b q=%b qbar=%b", $time, clk, t, q, qbar);
endmodule