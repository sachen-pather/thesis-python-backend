/*
 * Circuit: Vending Machine (wrong change)
 * Category: State Machines - Buggy
 * Complexity: COMPLEX
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_vending_machine(input wire clk, rst, input wire [1:0] coin, output reg dispense, output reg [1:0] change);
localparam S0=0, S5=1, S10=2, S15=3, S20=4, S25=5;
reg [2:0] state;

always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= S0;
        dispense <= 0;
        change <= 0;
    end else begin
        dispense <= 0;
        change <= 0;
        case (state)
            S0: begin
                case (coin)
                    2'b01: state <= S5;
                    2'b10: state <= S10;
                    2'b11: state <= S25;
                endcase
            end
            S5: begin
                case (coin)
                    2'b01: state <= S10;
                    2'b10: state <= S15;
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S10: begin
                case (coin)
                    2'b01: state <= S15;
                    2'b10: state <= S20;
                    // BUG: No change given
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S15: begin
                case (coin)
                    2'b01: state <= S20;
                    2'b10: state <= S25;
                    // BUG: No change given
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S20: begin
                case (coin)
                    2'b01: state <= S25;
                    2'b10: begin state <= S0; dispense <= 1; end
                    // BUG: No change given
                    2'b11: begin state <= S0; dispense <= 1; end
                endcase
            end
            S25: begin
                case (coin)
                    2'b01: begin state <= S0; dispense <= 1; end
                    2'b10: begin state <= S0; dispense <= 1; change <= 2'b01; end
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b10; end
                endcase
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst; reg [1:0] coin; wire dispense; wire [1:0] change;
bad_vending_machine dut(.clk(clk), .rst(rst), .coin(coin), .dispense(dispense), .change(change));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; coin=2'b00; #10; rst=0;
    coin=2'b10;#10; coin=2'b10;#10; coin=2'b10;#10; coin=2'b00;#10;
    coin=2'b11;#10; coin=2'b00;#10;
    coin=2'b11;#10; coin=2'b10;#10; coin=2'b00;#10;
    $finish;
end
initial $monitor("Time=%0t coin=%d state=%d dispense=%b change=%d", $time, coin, dut.state, dispense, change);
endmodule