/*
 * Circuit: Register File (no write)
 * Category: CPU Components - Buggy
 * Complexity: COMPLEX
 * Status: BUGGY
 * 
 * Extracted from test suite
 */

`timescale 1ns/1ps
module bad_register_file(input wire clk, we, input wire [1:0] rd_addr1, rd_addr2, wr_addr, 
                         input wire [7:0] wr_data, output wire [7:0] rd_data1, rd_data2);
reg [7:0] regs [0:3];

// BUG: Write enable ignored
always @(posedge clk) begin
    regs[wr_addr] <= wr_data;
end

assign rd_data1 = regs[rd_addr1];
assign rd_data2 = regs[rd_addr2];
endmodule

module testbench;
reg clk, we; reg [1:0] rd_addr1, rd_addr2, wr_addr; reg [7:0] wr_data;
wire [7:0] rd_data1, rd_data2;
bad_register_file dut(.clk(clk), .we(we), .rd_addr1(rd_addr1), .rd_addr2(rd_addr2), 
                      .wr_addr(wr_addr), .wr_data(wr_data), .rd_data1(rd_data1), .rd_data2(rd_data2));
initial begin clk=0; forever #5 clk=~clk; end
initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, testbench);
    we=0; rd_addr1=0; rd_addr2=0; wr_addr=0; wr_data=0; #10;
    we=1; wr_addr=0; wr_data=8'hAA; #10;
    we=1; wr_addr=1; wr_data=8'h55; #10;
    we=0; wr_addr=2; wr_data=8'hCC; #10; // Should NOT write
    we=0; rd_addr1=0; rd_addr2=1; #10;
    rd_addr1=2; rd_addr2=0; #10;
    $finish;
end
initial $monitor("Time=%0t we=%b wr_addr=%d wr_data=%h rd1_addr=%d rd1_data=%h rd2_addr=%d rd2_data=%h",
                 $time, we, wr_addr, wr_data, rd_addr1, rd_data1, rd_addr2, rd_data2);
endmodule