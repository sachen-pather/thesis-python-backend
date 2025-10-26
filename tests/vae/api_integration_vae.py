"""
API Integration: Final Hybrid VAE Verifier
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from final_hybrid_verification import verify_circuit_final
from services.simulation_service import SimulationService
import traceback


class HybridVerificationService:
    """Service for hybrid VAE + rule verification"""
    
    def __init__(self):
        self.sim_service = SimulationService()
        self.verification_available = self._check_vae_availability()
    
    def _check_vae_availability(self):
        """Check if VAE model is available"""
        try:
            from final_hybrid_verification import verify_circuit_final
            return True
        except Exception as e:
            print(f"VAE not available: {e}")
            return False
    
    def verify_circuit_comprehensive(self, verilog_code, session_id=None):
        """Run simulation + hybrid VAE verification"""
        result = {
            "success": False,
            "simulation": {},
            "vae_verification": {},
            "error": None
        }
        
        try:
            # Run simulation
            sim_success, stdout, stderr, error = self.sim_service.simulate_verilog(verilog_code)
            
            result["simulation"] = {
                "success": sim_success,
                "stdout": stdout,
                "stderr": stderr,
                "error": error
            }
            
            if not sim_success:
                result["error"] = f"Simulation failed: {error}"
                return result
            
            # Run hybrid VAE
            if self.verification_available:
                try:
                    is_anomalous, confidence, message = verify_circuit_final(verilog_code)
                    
                    result["vae_verification"] = {
                        "available": True,
                        "is_anomalous": is_anomalous,
                        "confidence": float(confidence),
                        "message": message
                    }
                except Exception as e:
                    result["vae_verification"] = {
                        "available": False,
                        "error": str(e)
                    }
            
            result["success"] = True
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result


# Test
if __name__ == "__main__":
    service = HybridVerificationService()
    
    test_verilog = '''`timescale 1ns/1ps
module and_gate(input wire a, b, output wire y);
assign y = a & b;
endmodule

module testbench;
    reg a, b; wire y;
    and_gate dut(.a(a), .b(b), .y(y));
    initial begin
        a = 0; b = 0; #10;
        a = 1; b = 1; #10;
        $finish;
    end
    initial $monitor("Time=%0t a=%b b=%b y=%b", $time, a, b, y);
endmodule'''
    
    print("Testing...")
    result = service.verify_circuit_comprehensive(test_verilog)
    
    if result["success"]:
        print("✓ Success")
        print(f"VAE: {result['vae_verification'].get('message')}")
    else:
        print(f"✗ Failed: {result['error']}")