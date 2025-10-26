# tests/integration/test_complete_flow.py
"""
Complete integration test for the entire Verilog generation and verification pipeline:
1. User prompt → Mermaid diagram
2. Mermaid → Verilog code
3. Verilog → Simulation
4. Waveform → LLM + VAE verification

This tests the full end-to-end flow including the new VAE verification.
"""

import sys
import os
import requests
import json
import time
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"test_{int(time.time())}"

class IntegrationTestSuite:
    def __init__(self):
        self.session_id = TEST_SESSION_ID
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message, status="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "ERROR": "❌",
            "WARNING": "⚠️"
        }.get(status, "📝")
        print(f"[{timestamp}] {status_emoji} {message}")
    
    def check_server_health(self):
        """Test 1: Check if server is running and healthy"""
        self.log("Testing server health...")
        
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=10)
            self.log(f"Health check response status: {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                self.log("Server is healthy", "SUCCESS")
                self.log(f"  - Version: {health_data.get('version', 'unknown')}")
                self.log(f"  - Multi-model: {health_data.get('services', {}).get('multi_model', False)}")
                self.log(f"  - iverilog: {health_data.get('services', {}).get('iverilog', False)}")
                return True
            else:
                self.log(f"Server health check failed: {response.status_code}", "ERROR")
                self.log(f"Response: {response.text}", "ERROR")
                return False
        except requests.exceptions.ConnectionError as e:
            self.log(f"Cannot connect to server at {BASE_URL}: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Health check error: {e}", "ERROR")
            return False
    
    def check_vae_availability(self):
        """Test 2: Check VAE verification system"""
        self.log("Testing VAE verification availability...")
        
        try:
            response = requests.get(f"{BASE_URL}/api/health/verification", timeout=10)
            if response.status_code == 200:
                verif_data = response.json()
                vae_status = verif_data.get("verification_systems", {}).get("vae", {})
                
                if vae_status.get("available", False):
                    self.log("VAE verification is available", "SUCCESS")
                    self.log(f"  - Threshold: {vae_status.get('threshold', 'unknown')}")
                    self.log(f"  - Device: {vae_status.get('device', 'unknown')}")
                    return True
                else:
                    self.log(f"VAE not available: {vae_status.get('error', 'unknown')}", "WARNING")
                    return False
            else:
                self.log("Could not check verification status", "WARNING")
                return False
        except Exception as e:
            self.log(f"Verification check failed: {e}", "WARNING")
            return False
    
    def test_mermaid_generation(self, prompt):
        """Test 3: Generate Mermaid diagram from prompt"""
        self.log(f"Testing Mermaid generation with prompt: '{prompt[:50]}...'")
        
        try:
            payload = {
                "prompt": prompt,
                "model": "groq",
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/design/generate-mermaid",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    mermaid_code = data.get("mermaid_code", "")
                    self.log("Mermaid generation successful", "SUCCESS")
                    self.log(f"  - Length: {len(mermaid_code)} characters")
                    self.log(f"  - Model: {data.get('model_used', 'unknown')}")
                    self.results['mermaid_code'] = mermaid_code
                    return mermaid_code
                else:
                    self.log(f"Mermaid generation failed: {data.get('error', 'unknown')}", "ERROR")
                    return None
            else:
                self.log(f"Mermaid API error: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"Mermaid generation exception: {e}", "ERROR")
            return None
    
    def test_verilog_generation(self, mermaid_code, description=""):
        """Test 4: Generate Verilog from Mermaid"""
        self.log("Testing Verilog generation from Mermaid...")
        
        try:
            payload = {
                "mermaid_code": mermaid_code,
                "description": description,
                "model": "groq",
                "use_rag": False,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/design/generate-verilog",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    verilog_code = data.get("verilog_code", "")
                    stats = data.get("stats", {})
                    self.log("Verilog generation successful", "SUCCESS")
                    self.log(f"  - Lines: {stats.get('lines', 0)}")
                    self.log(f"  - Modules: {stats.get('modules', 0)}")
                    self.log(f"  - Has testbench: {stats.get('has_testbench', False)}")
                    self.log(f"  - Validation issues: {len(data.get('validation_issues', []))}")
                    self.results['verilog_code'] = verilog_code
                    return verilog_code
                else:
                    self.log(f"Verilog generation failed: {data.get('error', 'unknown')}", "ERROR")
                    return None
            else:
                self.log(f"Verilog API error: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"Verilog generation exception: {e}", "ERROR")
            return None
    
    def test_simulation_with_verification(self, verilog_code):
        """Test 5: Run simulation with both LLM and VAE verification"""
        self.log("Testing simulation with comprehensive verification...")
        
        try:
            payload = {
                "verilog_code": verilog_code,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/simulation/run-with-verification",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.log("Simulation successful", "SUCCESS")
                    
                    # Check simulation results
                    sim_time = data.get("simulation_time", 0)
                    csv_length = len(data.get("waveform_csv", ""))
                    self.log(f"  - Simulation time: {sim_time}s")
                    self.log(f"  - Waveform CSV length: {csv_length} chars")
                    
                    # Check verification results
                    verification = data.get("verification", {})
                    assessment = data.get("assessment", {})
                    
                    # VAE Verification
                    vae_verif = verification.get("vae_verification", {})
                    if vae_verif.get("available"):
                        is_anomalous = vae_verif.get("is_anomalous", False)
                        error = vae_verif.get("error", 0)
                        confidence = vae_verif.get("confidence", "unknown")
                        
                        status = "ERROR" if is_anomalous else "SUCCESS"
                        self.log(f"VAE Verification: {vae_verif.get('message', 'unknown')}", status)
                        self.log(f"  - Anomalous: {is_anomalous}")
                        self.log(f"  - Error: {error:.6f}")
                        self.log(f"  - Confidence: {confidence}")
                    else:
                        self.log(f"VAE verification unavailable: {vae_verif.get('error', 'unknown')}", "WARNING")
                    
                    # LLM Verification
                    llm_verif = verification.get("llm_verification", {})
                    if llm_verif.get("available"):
                        self.log("LLM verification completed", "SUCCESS")
                        analysis = llm_verif.get("analysis", "")
                        self.log(f"  - Analysis length: {len(analysis)} chars")
                    else:
                        self.log(f"LLM verification unavailable: {llm_verif.get('error', 'unknown')}", "WARNING")
                    
                    # Overall Assessment
                    overall_status = assessment.get("overall_status", "unknown")
                    self.log(f"Overall Assessment: {overall_status}")
                    
                    self.results['simulation_success'] = True
                    self.results['verification'] = verification
                    self.results['assessment'] = assessment
                    
                    return data
                else:
                    self.log(f"Simulation failed: {data.get('error', 'unknown')}", "ERROR")
                    return None
            else:
                self.log(f"Simulation API error: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"Simulation exception: {e}", "ERROR")
            return None
    
    def test_vae_only_verification(self, verilog_code):
        """Test 6: Test standalone VAE verification endpoint"""
        self.log("Testing standalone VAE verification...")
        
        try:
            payload = {
                "verilog_code": verilog_code,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/verification/vae",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    vae_result = data.get("vae_verification", {})
                    self.log("Standalone VAE verification successful", "SUCCESS")
                    self.log(f"  - Is anomalous: {vae_result.get('is_anomalous', 'unknown')}")
                    self.log(f"  - Error: {vae_result.get('error', 0):.6f}")
                    self.log(f"  - Threshold: {vae_result.get('threshold', 0):.6f}")
                    return data
                else:
                    self.log(f"VAE verification failed: {data.get('error', 'unknown')}", "ERROR")
                    return None
            else:
                self.log(f"VAE API error: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"VAE verification exception: {e}", "ERROR")
            return None
    
    def run_complete_test(self):
        """Run the complete integration test suite"""
        self.log("="*60)
        self.log("STARTING COMPLETE SYSTEM INTEGRATION TEST")
        self.log("="*60)
        
        # Test prompts - mix of simple and complex
        test_cases = [
            {
                "name": "Simple Toggle Flip-Flop",
                "prompt": "Create a toggle flip-flop that changes output on each clock edge with reset",
                "description": "Basic sequential circuit test"
            },
            {
                "name": "4-bit Counter", 
                "prompt": "Design a 4-bit binary counter with clock and reset inputs",
                "description": "Counter circuit for VAE training validation"
            },
            {
                "name": "Basic Shift Register",
                "prompt": "Create a 4-bit shift register with serial input and parallel output",
                "description": "Shift register similar to VAE training data"
            }
        ]
        
        overall_success = True
        
        # Step 0: Check server health
        if not self.check_server_health():
            self.log("Server health check failed - aborting tests", "ERROR")
            return False
        
        # Step 0.5: Check VAE availability
        vae_available = self.check_vae_availability()
        if not vae_available:
            self.log("VAE not available - tests will continue but VAE verification will be skipped", "WARNING")
        
        # Run tests for each case
        for i, test_case in enumerate(test_cases, 1):
            self.log(f"\n{'='*40}")
            self.log(f"TEST CASE {i}: {test_case['name']}")
            self.log(f"{'='*40}")
            
            # Step 1: Generate Mermaid
            mermaid_code = self.test_mermaid_generation(test_case['prompt'])
            if not mermaid_code:
                self.log(f"Test case {i} failed at Mermaid generation", "ERROR")
                overall_success = False
                continue
            
            # Step 2: Generate Verilog
            verilog_code = self.test_verilog_generation(mermaid_code, test_case['description'])
            if not verilog_code:
                self.log(f"Test case {i} failed at Verilog generation", "ERROR")
                overall_success = False
                continue
            
            # Step 3: Simulate with verification
            sim_result = self.test_simulation_with_verification(verilog_code)
            if not sim_result:
                self.log(f"Test case {i} failed at simulation", "ERROR")
                overall_success = False
                continue
            
            # Standalone VAE test removed - API integration covers VAE verification
            
            self.log(f"Test case {i} completed successfully", "SUCCESS")
        
        # Final summary
        self.log(f"\n{'='*60}")
        self.log("INTEGRATION TEST SUMMARY")
        self.log(f"{'='*60}")
        
        total_time = time.time() - self.start_time
        self.log(f"Total execution time: {total_time:.2f} seconds")
        self.log(f"Session ID: {self.session_id}")
        self.log(f"VAE available: {vae_available}")
        
        if overall_success:
            self.log("ALL INTEGRATION TESTS PASSED", "SUCCESS")
        else:
            self.log("SOME INTEGRATION TESTS FAILED", "ERROR")
        
        return overall_success
    
    def run_focused_vae_test(self):
        """Run a focused test specifically for VAE verification"""
        self.log("="*60)
        self.log("FOCUSED VAE VERIFICATION TEST")
        self.log("="*60)
        
        # Use a simple toggle FF that should be similar to training data
        test_verilog = '''`timescale 1ns/1ps
module toggle_ff(input wire clk, rst_n, output reg q);
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 1'b0;
    else q <= ~q;
end
endmodule

module testbench;
    reg clk, rst_n; wire q;
    toggle_ff dut(.clk(clk), .rst_n(rst_n), .q(q));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst_n = 0;
        #20 rst_n = 1;
        #200 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b q=%b", $time, clk, q);
endmodule'''
        
        self.log("Testing with known good Verilog (toggle flip-flop)...")
        
        # Test simulation + verification
        sim_result = self.test_simulation_with_verification(test_verilog)
        
        # Test standalone VAE
        vae_result = self.test_vae_only_verification(test_verilog)
        
        if sim_result and vae_result:
            self.log("VAE verification test completed successfully", "SUCCESS")
            return True
        else:
            self.log("VAE verification test failed", "ERROR")
            return False


def main():
    """Main test execution"""
    test_suite = IntegrationTestSuite()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "vae-only":
        # Run focused VAE test
        success = test_suite.run_focused_vae_test()
    else:
        # Run complete integration test
        success = test_suite.run_complete_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()