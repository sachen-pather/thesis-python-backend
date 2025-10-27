`timescale 1ns/1ps

// 4-bit comparator for machine learning training data
module comparator_4bit(
    input wire [3:0] a, b,
    output reg eq, gt, lt
);
    // Behavioral implementation
    always @(*) begin
        if (a == b) begin
            eq = 1; gt = 0; lt = 0;
        end else if (a > b) begin
            eq = 0; gt = 1; lt = 0;
        end else begin
            eq = 0; gt = 0; lt = 1;
        end
    end
endmodule

module testbench;
    reg [3:0] a, b;
    wire eq, gt, lt;
    
    comparator_4bit dut(
        .a(a),
        .b(b),
        .eq(eq),
        .gt(gt),
        .lt(lt)
    );
    
    initial begin
        // Test various comparisons
        a = 4'b0101; b = 4'b0101; #10;  // 5 == 5
        a = 4'b1000; b = 4'b0011; #10;  // 8 > 3
        a = 4'b0010; b = 4'b0111; #10;  // 2 < 7
        a = 4'b1111; b = 4'b1111; #10;  // 15 == 15
        a = 4'b1010; b = 4'b0101; #10;  // 10 > 5
        a = 4'b0001; b = 4'b1110; #10;  // 1 < 14
        a = 4'b0000; b = 4'b0000; #10;  // 0 == 0
        $finish;
    end
    
    initial $monitor("Time=%0t a=%b(%d) b=%b(%d) eq=%b gt=%b lt=%b", 
                     $time, a, a, b, b, eq, gt, lt);
endmodule
