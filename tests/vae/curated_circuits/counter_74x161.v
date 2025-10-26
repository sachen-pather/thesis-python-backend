`timescale 1ns/1ps

// 4-bit synchronous counter (74x161 style) for machine learning training data
module counter_74x161(
    input wire clr_bar,              // Async clear (active low)
    input wire ld_bar,               // Load (active low)
    input wire ent,                  // Enable T
    input wire enp,                  // Enable P
    input wire clk,                  // Clock
    input wire a, b, c, d,           // Data inputs
    output reg qa, qb, qc, qd,       // Data outputs
    output wire rco                  // Ripple carry output
);
    // Internal signals
    wire ld = ~ld_bar;
    wire count_enable = ent & enp;
    
    // Ripple carry output - high when all outputs high and ent is high
    assign rco = ent & qa & qb & qc & qd;
    
    always @(posedge clk or negedge clr_bar) begin
        if (~clr_bar) begin
            // Asynchronous clear
            qa <= 1'b0;
            qb <= 1'b0;
            qc <= 1'b0;
            qd <= 1'b0;
        end else if (ld) begin
            // Synchronous load
            qa <= a;
            qb <= b;
            qc <= c;
            qd <= d;
        end else if (count_enable) begin
            // Count up
            {qd, qc, qb, qa} <= {qd, qc, qb, qa} + 1;
        end
        // If count_enable is low, hold current value
    end
endmodule

module testbench;
    reg clr_bar, ld_bar, ent, enp, clk;
    reg a, b, c, d;
    wire qa, qb, qc, qd, rco;
    
    counter_74x161 dut(
        .clr_bar(clr_bar),
        .ld_bar(ld_bar),
        .ent(ent),
        .enp(enp),
        .clk(clk),
        .a(a), .b(b), .c(c), .d(d),
        .qa(qa), .qb(qb), .qc(qc), .qd(qd),
        .rco(rco)
    );
    
    initial begin
        // Initialize
        clk = 0;
        clr_bar = 0; ld_bar = 1; ent = 1; enp = 1;
        a = 0; b = 0; c = 0; d = 0;
        #10;
        
        // Release clear and start counting
        clr_bar = 1;
        #80;  // Count for a while
        
        // Test load function - load value 1010
        a = 0; b = 1; c = 0; d = 1;
        ld_bar = 0; #10;  // Load
        ld_bar = 1; #30;  // Continue counting from loaded value
        
        // Test clear again
        clr_bar = 0; #10;
        clr_bar = 1; #20;
        
        $finish;
    end
    
    // Clock generation - 10ns period
    always #5 clk = ~clk;
    
    initial $monitor("Time=%0t clr_bar=%b ld_bar=%b ent=%b enp=%b qa=%b qb=%b qc=%b qd=%b rco=%b count=%d", 
                     $time, clr_bar, ld_bar, ent, enp, qa, qb, qc, qd, rco, {qd,qc,qb,qa});
endmodule