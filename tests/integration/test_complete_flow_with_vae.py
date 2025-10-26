"""
Complete End-to-End Integration Test
Tests the full pipeline: Prompt → Mermaid → Verilog → Simulation → Hybrid VAE + LLM Verification

Run this from your project root:
    python tests/integration/test_complete_flow_with_vae.py
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

# Configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"e2e_test_{int(time.time())}"

class EndToEndTest:
    def __init__(self):
        self.session_id = TEST_SESSION_ID
        self.results = {}
        self.start_time = time.time()
        self.test_cases = []
        
    def log(self, message, status="INFO"):
        """Log test progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "TEST": "🧪"
        }.get(status, "📝")
        print(f"[{timestamp}] {emoji} {message}")
    
    def check_server_health(self):
        """Test 0: Verify server is running"""
        self.log("=" * 80)
        self.log("TEST 0: Server Health Check", "TEST")
        self.log("=" * 80)
        
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log(f"Server Status: {data.get('status', 'unknown')}", "SUCCESS")
                self.log(f"Version: {data.get('version', 'unknown')}")
                
                services = data.get('services', {})
                self.log(f"Groq API: {'✓' if services.get('groq_api') else '✗'}")
                self.log(f"iVerilog: {'✓' if services.get('iverilog') else '✗'}")
                self.log(f"Multi-Model: {'✓' if services.get('multi_model') else '✗'}")
                
                return True
            else:
                self.log(f"Health check failed: {response.status_code}", "ERROR")
                return False
                
        except requests.exceptions.ConnectionError:
            self.log(f"Cannot connect to server at {BASE_URL}", "ERROR")
            self.log("Make sure server is running: python main.py", "WARNING")
            return False
        except Exception as e:
            self.log(f"Health check error: {e}", "ERROR")
            return False
    
    def check_vae_availability(self):
        """Test 0.5: Check VAE verification system"""
        self.log("\n" + "=" * 80)
        self.log("TEST 0.5: VAE Verification System Check", "TEST")
        self.log("=" * 80)
        
        try:
            response = requests.get(f"{BASE_URL}/api/health/verification", timeout=10)
            if response.status_code == 200:
                data = response.json()
                vae = data.get("verification_systems", {}).get("vae", {})
                
                if vae.get("available"):
                    self.log("VAE System: AVAILABLE", "SUCCESS")
                    self.log(f"  Threshold: {vae.get('threshold', 'unknown')}")
                    self.log(f"  Device: {vae.get('device', 'unknown')}")
                    self.log(f"  Model: Hybrid (Rules + VAE)")
                    return True
                else:
                    self.log(f"VAE System: UNAVAILABLE - {vae.get('error', 'unknown')}", "WARNING")
                    return False
            else:
                self.log("Could not check VAE status", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"VAE check error: {e}", "WARNING")
            return False
    
    def test_mermaid_generation(self, prompt, test_name):
        """Test 1: Generate Mermaid diagram"""
        self.log(f"\nGenerating Mermaid for: {test_name}")
        
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
                    mermaid = data.get("mermaid_code", "")
                    self.log(f"  Mermaid: {len(mermaid)} chars", "SUCCESS")
                    return mermaid
                else:
                    self.log(f"  Failed: {data.get('error')}", "ERROR")
                    return None
            else:
                self.log(f"  API Error: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"  Exception: {e}", "ERROR")
            return None
    
    def test_verilog_generation(self, mermaid_code, description, test_name):
        """Test 2: Generate Verilog from Mermaid"""
        self.log(f"Generating Verilog for: {test_name}")
        
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
                    verilog = data.get("verilog_code", "")
                    stats = data.get("stats", {})
                    issues = data.get("validation_issues", [])
                    
                    self.log(f"  Verilog: {stats.get('lines', 0)} lines", "SUCCESS")
                    self.log(f"  Modules: {stats.get('modules', 0)}")
                    self.log(f"  Has testbench: {stats.get('has_testbench', False)}")
                    
                    if issues:
                        self.log(f"  Validation issues: {len(issues)}", "WARNING")
                    
                    return verilog
                else:
                    self.log(f"  Failed: {data.get('error')}", "ERROR")
                    return None
            else:
                self.log(f"  API Error: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"  Exception: {e}", "ERROR")
            return None
    
    def test_simulation_with_vae(self, verilog_code, test_name, expected_anomalous=None):
        """Test 3: Run simulation with hybrid VAE verification"""
        self.log(f"Running simulation + VAE for: {test_name}")
        
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
                
                # Simulation results
                sim_success = data.get("success", False)
                sim_time = data.get("simulation_time", 0)
                waveform = data.get("waveform_csv", "")
                
                self.log(f"  Simulation: {'✓' if sim_success else '✗'}")
                self.log(f"  Time: {sim_time:.3f}s")
                self.log(f"  Waveform: {len(waveform)} chars")
                
                # VAE Verification results
                verification = data.get("verification", {})
                vae = verification.get("vae_verification", {})
                
                if vae.get("available"):
                    is_anom = vae.get("is_anomalous", False)
                    confidence = vae.get("confidence", 0)
                    message = vae.get("message", "")
                    
                    status = "ERROR" if is_anom else "SUCCESS"
                    self.log(f"  VAE Verdict: {'🔴 ANOMALOUS' if is_anom else '✓ NORMAL'}", status)
                    self.log(f"  Confidence: {confidence:.3f}")
                    self.log(f"  Message: {message[:60]}...")
                    
                    # Check against expected result
                    if expected_anomalous is not None:
                        correct = (is_anom == expected_anomalous)
                        if correct:
                            self.log(f"  Prediction: ✓ CORRECT", "SUCCESS")
                        else:
                            expected_str = "ANOMALOUS" if expected_anomalous else "NORMAL"
                            self.log(f"  Prediction: ✗ WRONG (expected {expected_str})", "ERROR")
                        
                        return {
                            "success": True,
                            "simulation_success": sim_success,
                            "vae_correct": correct,
                            "is_anomalous": is_anom,
                            "confidence": confidence
                        }
                else:
                    self.log(f"  VAE: Unavailable - {vae.get('error', 'unknown')}", "WARNING")
                
                return {
                    "success": sim_success,
                    "simulation_success": sim_success,
                    "vae_available": vae.get("available", False)
                }
                
            else:
                self.log(f"  API Error: {response.status_code}", "ERROR")
                return {"success": False}
                
        except Exception as e:
            self.log(f"  Exception: {e}", "ERROR")
            return {"success": False}
    
    def run_complete_test_case(self, prompt, description, test_name, expected_anomalous=None):
        """Run complete end-to-end test for one circuit"""
        self.log("\n" + "=" * 80)
        self.log(f"TEST CASE: {test_name}", "TEST")
        self.log("=" * 80)
        self.log(f"Prompt: {prompt[:60]}...")
        
        result = {
            "name": test_name,
            "prompt": prompt,
            "success": False,
            "steps_completed": []
        }
        
        # Step 1: Generate Mermaid
        mermaid = self.test_mermaid_generation(prompt, test_name)
        if not mermaid:
            self.log("❌ Test failed at Mermaid generation", "ERROR")
            return result
        result["steps_completed"].append("mermaid")
        result["mermaid"] = mermaid
        
        # Step 2: Generate Verilog
        verilog = self.test_verilog_generation(mermaid, description, test_name)
        if not verilog:
            self.log("❌ Test failed at Verilog generation", "ERROR")
            return result
        result["steps_completed"].append("verilog")
        result["verilog"] = verilog
        
        # Step 3: Simulate + VAE
        sim_result = self.test_simulation_with_vae(verilog, test_name, expected_anomalous)
        if not sim_result.get("success"):
            self.log("❌ Test failed at simulation", "ERROR")
            return result
        
        result["steps_completed"].append("simulation")
        result["steps_completed"].append("vae_verification")
        result["simulation_result"] = sim_result
        result["success"] = True
        
        if expected_anomalous is not None:
            result["vae_correct"] = sim_result.get("vae_correct", False)
        
        self.log(f"✅ Test case completed: {len(result['steps_completed'])} steps", "SUCCESS")
        return result
    
    def run_all_tests(self):
        """Run complete end-to-end test suite"""
        self.log("\n" + "=" * 80)
        self.log("COMPLETE END-TO-END SYSTEM TEST", "TEST")
        self.log("=" * 80)
        self.log(f"Session ID: {self.session_id}")
        self.log(f"Base URL: {BASE_URL}")
        
        # Step 0: Health checks
        if not self.check_server_health():
            self.log("\n❌ Server health check failed - aborting tests", "ERROR")
            return False
        
        vae_available = self.check_vae_availability()
        if not vae_available:
            self.log("\n⚠️  VAE not available - tests will continue without VAE verification", "WARNING")
        
        # Define test cases
        test_cases = [
            {
                "name": "Simple AND Gate (Good)",
                "prompt": "Create a 2-input AND gate",
                "description": "Basic combinational logic gate",
                "expected_anomalous": False
            },
            {
                "name": "Toggle Flip-Flop (Good)",
                "prompt": "Design a toggle flip-flop with clock and reset",
                "description": "Sequential circuit that toggles output on each clock",
                "expected_anomalous": False
            },
            {
                "name": "4-bit Counter (Good)",
                "prompt": "Create a 4-bit binary counter with clock and reset",
                "description": "Sequential counter similar to VAE training data",
                "expected_anomalous": False
            }
        ]
        
        # Run all test cases
        for test_case in test_cases:
            result = self.run_complete_test_case(
                test_case["prompt"],
                test_case["description"],
                test_case["name"],
                test_case.get("expected_anomalous")
            )
            self.test_cases.append(result)
            time.sleep(1)  # Brief pause between tests
        
        # Generate summary
        self.generate_summary()
        
        return True
    
    def generate_summary(self):
        """Generate test summary"""
        self.log("\n" + "=" * 80)
        self.log("END-TO-END TEST SUMMARY", "TEST")
        self.log("=" * 80)
        
        total_tests = len(self.test_cases)
        successful_tests = sum(1 for t in self.test_cases if t.get("success"))
        
        self.log(f"Total Test Cases: {total_tests}")
        self.log(f"Successful: {successful_tests}")
        self.log(f"Failed: {total_tests - successful_tests}")
        self.log(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
        
        # Step completion analysis
        all_steps = ["mermaid", "verilog", "simulation", "vae_verification"]
        step_completion = {step: 0 for step in all_steps}
        
        for test in self.test_cases:
            for step in test.get("steps_completed", []):
                if step in step_completion:
                    step_completion[step] += 1
        
        self.log("\nStep Completion:")
        for step, count in step_completion.items():
            self.log(f"  {step}: {count}/{total_tests} ({count/total_tests*100:.1f}%)")
        
        # VAE accuracy (if applicable)
        vae_predictions = [t for t in self.test_cases if "vae_correct" in t]
        if vae_predictions:
            vae_correct = sum(1 for t in vae_predictions if t.get("vae_correct"))
            self.log(f"\nVAE Accuracy: {vae_correct}/{len(vae_predictions)} ({vae_correct/len(vae_predictions)*100:.1f}%)")
        
        # Individual results
        self.log("\nIndividual Test Results:")
        for i, test in enumerate(self.test_cases, 1):
            status = "✅" if test.get("success") else "❌"
            steps = len(test.get("steps_completed", []))
            self.log(f"  {status} Test {i}: {test['name']} ({steps}/4 steps)")
            
            if "vae_correct" in test:
                vae_status = "✓" if test["vae_correct"] else "✗"
                self.log(f"      VAE: {vae_status}")
        
        # Total time
        total_time = time.time() - self.start_time
        self.log(f"\nTotal Test Time: {total_time:.2f}s")
        
        # Final verdict
        if successful_tests == total_tests:
            self.log("\n🎉 ALL TESTS PASSED!", "SUCCESS")
        elif successful_tests > 0:
            self.log(f"\n⚠️  {successful_tests}/{total_tests} tests passed", "WARNING")
        else:
            self.log("\n❌ ALL TESTS FAILED", "ERROR")
        
        self.log("=" * 80)


def main():
    """Main test execution"""
    print("\n🧪 Starting Complete End-to-End System Test\n")
    
    test_suite = EndToEndTest()
    success = test_suite.run_all_tests()
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()