import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.verilogeval_service import VerilogEvalService

class TestVerilogEvalService:
    def setup_method(self):
        self.service = VerilogEvalService()
    
    def test_syntax_compliance_basic(self):
        """Test basic syntax compliance evaluation"""
        valid_code = """
        module counter(
            input wire clk,
            input wire rst_n,
            output reg [3:0] count
        );
        
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                count <= 4'b0000;
            else
                count <= count + 1'b1;
        end
        
        endmodule
        """
        
        result = self.service._evaluate_syntax_compliance(valid_code)
        assert result['score'] > 15  # Should get decent score
        assert len(result['issues']) == 0  # No critical issues
    
    def test_complexity_assessment(self):
        """Test design complexity assessment"""
        simple_code = "module test(); endmodule"
        complex_code = """
        module complex_fsm(
            input clk, rst, start,
            output reg [2:0] state,
            output reg done
        );
        
        parameter IDLE = 0, ACTIVE = 1, DONE = 2;
        
        always @(posedge clk) begin
            case (state)
                IDLE: if (start) state <= ACTIVE;
                ACTIVE: state <= DONE;
                DONE: state <= IDLE;
            endcase
        end
        
        endmodule
        """
        
        simple_rating = self.service._assess_complexity(simple_code, "simple module")
        complex_rating = self.service._assess_complexity(complex_code, "FSM with states")
        
        assert simple_rating == "Basic"
        assert complex_rating in ["Intermediate", "Advanced"]
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality"""
        counter_code = "module counter(); reg [3:0] count; endmodule"
        result = self.service.get_benchmark_comparison(counter_code, "4-bit counter")
        
        assert 'problem_type' in result
        assert 'pattern_match_score' in result
        assert 'similar_hdlbits_problems' in result

if __name__ == "__main__":
    pytest.main([__file__])
