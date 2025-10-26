/*
 * Circuit: Vending Machine FSM
 * Category: State Machines - Normal
 * Complexity: COMPLEX
 * Status: NORMAL
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module vending_machine(input wire clk, rst, input wire [1:0] coin, output reg dispense, output reg [1:0] change);
// coin: 00=none, 01=5cent, 10=10cent, 11=25cent
// Item costs 30 cents
localparam S0=0, S5=1, S10=2, S15=3, S20=4, S25=5, S30=6;
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
                    2'b11: begin state <= S0; dispense <= 1; end // 30 cents
                endcase
            end
            S10: begin
                case (coin)
                    2'b01: state <= S15;
                    2'b10: state <= S20;
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b01; end // 35->30+5
                endcase
            end
            S15: begin
                case (coin)
                    2'b01: state <= S20;
                    2'b10: state <= S25;
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b10; end // 40->30+10
                endcase
            end
            S20: begin
                case (coin)
                    2'b01: state <= S25;
                    2'b10: begin state <= S0; dispense <= 1; end // 30 cents
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b11; end // 45->30+15
                endcase
            end
            S25: begin
                case (coin)
                    2'b01: begin state <= S0; dispense <= 1; end // 30 cents
                    2'b10: begin state <= S0; dispense <= 1; change <= 2'b01; end // 35->30+5
                    2'b11: begin state <= S0; dispense <= 1; change <= 2'b10; end // 50->30+20
                endcase
            end
        endcase
    end
end
endmodule

module testbench;
reg clk, rst; reg [1:0] coin; wire dispense; wire [1:0] change;
vending_machine dut(.clk(clk), .rst(rst), .coin(coin), .dispense(dispense), .change(change));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    rst=1; coin=2'b00; #10; rst=0;
    coin=2'b10;#10; coin=2'b10;#10; coin=2'b10;#10; coin=2'b00;#10; // 10+10+10=30
    coin=2'b11;#10; coin=2'b00;#10; // 25 + 5 previous = 30
    coin=2'b11;#10; coin=2'b10;#10; coin=2'b00;#10; // 25+10=35 with change
    $finish;
end
initial $monitor("Time=%0t coin=%d state=%d dispense=%b change=%d", $time, coin, dut.state, dispense, change);
endmodule