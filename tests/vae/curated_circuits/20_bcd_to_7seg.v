`timescale 1ns/1ps

// BCD to 7-segment decoder for machine learning training data
// Segments: {a, b, c, d, e, f, g} (active high)
module bcd_to_7seg(
    input wire [3:0] bcd,
    output reg [6:0] seg
);
    // Segment mapping (a, b, c, d, e, f, g)
    always @(*) begin
        case(bcd)
            4'b0000: seg = 7'b1111110;  // 0
            4'b0001: seg = 7'b0110000;  // 1
            4'b0010: seg = 7'b1101101;  // 2
            4'b0011: seg = 7'b1111001;  // 3
            4'b0100: seg = 7'b0110011;  // 4
            4'b0101: seg = 7'b1011011;  // 5
            4'b0110: seg = 7'b1011111;  // 6
            4'b0111: seg = 7'b1110000;  // 7
            4'b1000: seg = 7'b1111111;  // 8
            4'b1001: seg = 7'b1111011;  // 9
            default: seg = 7'b0000000;  // Blank for invalid BCD
        endcase
    end
endmodule

module testbench;
    reg [3:0] bcd;
    wire [6:0] seg;
    
    bcd_to_7seg dut(
        .bcd(bcd),
        .seg(seg)
    );
    
    initial begin
        // Test all valid BCD values (0-9)
        bcd = 4'b0000; #10;  // Display 0
        bcd = 4'b0001; #10;  // Display 1
        bcd = 4'b0010; #10;  // Display 2
        bcd = 4'b0011; #10;  // Display 3
        bcd = 4'b0100; #10;  // Display 4
        bcd = 4'b0101; #10;  // Display 5
        bcd = 4'b0110; #10;  // Display 6
        bcd = 4'b0111; #10;  // Display 7
        bcd = 4'b1000; #10;  // Display 8
        bcd = 4'b1001; #10;  // Display 9
        bcd = 4'b1010; #10;  // Invalid (blank)
        bcd = 4'b1111; #10;  // Invalid (blank)
        $finish;
    end
    
    initial $monitor("Time=%0t bcd=%b(%d) seg=%b", $time, bcd, bcd, seg);
endmodule
