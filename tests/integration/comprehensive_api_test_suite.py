"""
Comprehensive API Integration Test Suite
Tests the complete Verilog generation pipeline via REST API with diverse circuit patterns

Run from your project root:
    python tests/integration/comprehensive_api_test_suite.py
"""

import sys
import os
import requests
import json
import time
from datetime import datetime
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"comprehensive_test_{int(time.time())}"

# Try to import sklearn for better metrics
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not available - using manual metrics")


class ComprehensiveAPITestSuite:
    """Comprehensive test suite for VAE verification via API"""
    
    def __init__(self):
        self.session_id = TEST_SESSION_ID
        self.results = []
        self.start_time = time.time()
        self.test_cases = []
        
    def log(self, message, status="INFO"):
        """Log test progress with emojis"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "TEST": "🧪",
            "CATEGORY": "📁"
        }.get(status, "📝")
        print(f"[{timestamp}] {emoji} {message}")
    
    def check_server_health(self):
        """Verify server is running and VAE is available"""
        self.log("=" * 80)
        self.log("SERVER HEALTH CHECK", "TEST")
        self.log("=" * 80)
        
        try:
            # Check main health
            response = requests.get(f"{BASE_URL}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log(f"Server Status: {data.get('status', 'unknown')}", "SUCCESS")
                self.log(f"Version: {data.get('version', 'unknown')}")
                
                services = data.get('services', {})
                self.log(f"Groq API: {'✓' if services.get('groq_api') else '✗'}")
                self.log(f"iVerilog: {'✓' if services.get('iverilog') else '✗'}")
            else:
                self.log(f"Health check failed: {response.status_code}", "ERROR")
                return False
            
            # Check VAE verification system
            response = requests.get(f"{BASE_URL}/api/health/verification", timeout=10)
            if response.status_code == 200:
                data = response.json()
                vae = data.get("verification_systems", {}).get("vae", {})
                
                if vae.get("available"):
                    self.log("VAE System: AVAILABLE", "SUCCESS")
                    self.log(f"  Threshold: {vae.get('threshold', 'unknown')}")
                    self.log(f"  Device: {vae.get('device', 'unknown')}")
                    return True
                else:
                    self.log(f"VAE System: UNAVAILABLE - {vae.get('error', 'unknown')}", "ERROR")
                    return False
            else:
                self.log("Could not check VAE status", "WARNING")
                return False
                
        except requests.exceptions.ConnectionError:
            self.log(f"Cannot connect to server at {BASE_URL}", "ERROR")
            self.log("Make sure server is running: python main.py", "WARNING")
            return False
        except Exception as e:
            self.log(f"Health check error: {e}", "ERROR")
            return False
    
    def create_test_dataset(self):
        """Create comprehensive test dataset with diverse circuit types"""
        
        # ========== LOGIC GATES ==========
        logic_gates = [
            {
                "name": "2-Input AND Gate",
                "prompt": "Create a 2-input AND gate",
                "category": "Logic Gates",
                "expected_anomalous": False
            },
            {
                "name": "2-Input OR Gate",
                "prompt": "Design a 2-input OR gate",
                "category": "Logic Gates",
                "expected_anomalous": False
            },
            {
                "name": "2-Input XOR Gate",
                "prompt": "Build a 2-input XOR gate",
                "category": "Logic Gates",
                "expected_anomalous": False
            },
            {
                "name": "NOT Gate (Inverter)",
                "prompt": "Create an inverter (NOT gate)",
                "category": "Logic Gates",
                "expected_anomalous": False
            },
            {
                "name": "2-Input NAND Gate",
                "prompt": "Design a 2-input NAND gate",
                "category": "Logic Gates",
                "expected_anomalous": False
            },
            {
                "name": "2-Input NOR Gate",
                "prompt": "Build a 2-input NOR gate",
                "category": "Logic Gates",
                "expected_anomalous": False
            },
        ]
        
        # ========== MULTIPLEXERS ==========
        multiplexers = [
            {
                "name": "2:1 Multiplexer",
                "prompt": "Create a 2-to-1 multiplexer with select signal",
                "category": "Multiplexers",
                "expected_anomalous": False
            },
            {
                "name": "4:1 Multiplexer",
                "prompt": "Design a 4-to-1 multiplexer with 2-bit select",
                "category": "Multiplexers",
                "expected_anomalous": False
            },
        ]
        
        # ========== DECODERS/ENCODERS ==========
        decoders = [
            {
                "name": "2:4 Decoder",
                "prompt": "Create a 2-to-4 decoder with enable signal",
                "category": "Decoders/Encoders",
                "expected_anomalous": False
            },
            {
                "name": "4:2 Priority Encoder",
                "prompt": "Design a 4-to-2 priority encoder",
                "category": "Decoders/Encoders",
                "expected_anomalous": False
            },
        ]
        
        # ========== ARITHMETIC ==========
        arithmetic = [
            {
                "name": "Half Adder",
                "prompt": "Create a half adder circuit",
                "category": "Arithmetic",
                "expected_anomalous": False
            },
            {
                "name": "Full Adder",
                "prompt": "Design a full adder with carry in and carry out",
                "category": "Arithmetic",
                "expected_anomalous": False
            },
            {
                "name": "4-bit Ripple Carry Adder",
                "prompt": "Build a 4-bit ripple carry adder",
                "category": "Arithmetic",
                "expected_anomalous": False
            },
            {
                "name": "2-bit Comparator",
                "prompt": "Create a 2-bit magnitude comparator with equal, greater, less outputs",
                "category": "Arithmetic",
                "expected_anomalous": False
            },
        ]
        
        # ========== SEQUENTIAL - FLIP FLOPS ==========
        flip_flops = [
            {
                "name": "D Flip-Flop",
                "prompt": "Design a D flip-flop with clock and reset",
                "category": "Sequential - Flip Flops",
                "expected_anomalous": False
            },
            {
                "name": "T Flip-Flop",
                "prompt": "Create a T (toggle) flip-flop with clock and reset",
                "category": "Sequential - Flip Flops",
                "expected_anomalous": False
            },
            {
                "name": "JK Flip-Flop",
                "prompt": "Build a JK flip-flop with clock and reset",
                "category": "Sequential - Flip Flops",
                "expected_anomalous": False
            },
        ]
        
        # ========== SEQUENTIAL - COUNTERS ==========
        counters = [
            {
                "name": "4-bit Up Counter",
                "prompt": "Create a 4-bit binary up counter with clock and reset",
                "category": "Sequential - Counters",
                "expected_anomalous": False
            },
            {
                "name": "4-bit Down Counter",
                "prompt": "Design a 4-bit binary down counter with clock and reset",
                "category": "Sequential - Counters",
                "expected_anomalous": False
            },
            {
                "name": "4-bit Up-Down Counter",
                "prompt": "Build a 4-bit up-down counter with direction control",
                "category": "Sequential - Counters",
                "expected_anomalous": False
            },
            {
                "name": "Mod-10 Counter (BCD)",
                "prompt": "Create a modulo-10 BCD counter that counts 0-9 and resets",
                "category": "Sequential - Counters",
                "expected_anomalous": False
            },
        ]
        
        # ========== SEQUENTIAL - SHIFT REGISTERS ==========
        shift_registers = [
            {
                "name": "4-bit SISO Shift Register",
                "prompt": "Design a 4-bit serial-in serial-out shift register",
                "category": "Sequential - Shift Registers",
                "expected_anomalous": False
            },
            {
                "name": "4-bit PISO Shift Register",
                "prompt": "Create a 4-bit parallel-in serial-out shift register",
                "category": "Sequential - Shift Registers",
                "expected_anomalous": False
            },
        ]
        
        # ========== STATE MACHINES ==========
        state_machines = [
            {
                "name": "2-State Toggle FSM",
                "prompt": "Design a simple 2-state FSM that toggles between states on trigger",
                "category": "State Machines",
                "expected_anomalous": False
            },
            {
                "name": "Sequence Detector (101)",
                "prompt": "Create a Moore FSM to detect the sequence 101 in serial input",
                "category": "State Machines",
                "expected_anomalous": False
            },
        ]
        
        # ========== MEMORIES ==========
        memories = [
            {
                "name": "4x8 Register File",
                "prompt": "Design a 4-word by 8-bit register file with read and write",
                "category": "Memory",
                "expected_anomalous": False
            },
        ]
        
        # Combine all test cases
        self.test_cases = (
            logic_gates + 
            multiplexers + 
            decoders + 
            arithmetic + 
            flip_flops + 
            counters + 
            shift_registers + 
            state_machines + 
            memories
        )
        
        self.log(f"Created test dataset: {len(self.test_cases)} test cases", "SUCCESS")
        
        # Print category breakdown
        categories = {}
        for test in self.test_cases:
            cat = test['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        self.log("\nCategory breakdown:", "CATEGORY")
        for cat, count in sorted(categories.items()):
            self.log(f"  {cat:30s}: {count:2d} tests")
    
    def run_single_test(self, test_case):
        """Run a single end-to-end test case"""
        test_name = test_case['name']
        
        try:
            # Step 1: Generate Mermaid
            mermaid_payload = {
                "prompt": test_case['prompt'],
                "model": "claude",
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/design/generate-mermaid",
                json=mermaid_payload,
                timeout=30
            )
            
            if response.status_code != 200 or not response.json().get("success"):
                return {
                    'name': test_name,
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected_anomalous'],
                    'success': False,
                    'stage_failed': 'mermaid',
                    'error': response.json().get('error', 'Unknown error')
                }
            
            mermaid_code = response.json()['mermaid_code']
            
            # Step 2: Generate Verilog
            verilog_payload = {
                "mermaid_code": mermaid_code,
                "description": test_case['prompt'],
                "model": "claude",
                "use_rag": False,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/design/generate-verilog",
                json=verilog_payload,
                timeout=30
            )
            
            if response.status_code != 200 or not response.json().get("success"):
                return {
                    'name': test_name,
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected_anomalous'],
                    'success': False,
                    'stage_failed': 'verilog',
                    'error': response.json().get('error', 'Unknown error')
                }
            
            verilog_code = response.json()['verilog_code']
            
            # Step 3: Simulate with VAE verification
            sim_payload = {
                "verilog_code": verilog_code,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/simulation/run-with-verification",
                json=sim_payload,
                timeout=60
            )
            
            if response.status_code != 200:
                return {
                    'name': test_name,
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected_anomalous'],
                    'success': False,
                    'stage_failed': 'simulation',
                    'error': f"API error: {response.status_code}"
                }
            
            data = response.json()
            
            # Extract results
            sim_success = data.get("success", False)
            verification = data.get("verification", {})
            vae = verification.get("vae_verification", {})
            
            if not sim_success:
                return {
                    'name': test_name,
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected_anomalous'],
                    'success': False,
                    'stage_failed': 'simulation',
                    'error': data.get('error', 'Simulation failed')
                }
            
            if not vae.get("available"):
                return {
                    'name': test_name,
                    'category': test_case['category'],
                    'expected_anomalous': test_case['expected_anomalous'],
                    'success': False,
                    'stage_failed': 'vae',
                    'error': vae.get('error', 'VAE not available')
                }
            
            # VAE prediction
            is_anomalous = vae.get("is_anomalous", False)
            confidence = vae.get("confidence", 0.0)
            message = vae.get("message", "")
            
            # Check correctness
            correct = (is_anomalous == test_case['expected_anomalous'])
            
            return {
                'name': test_name,
                'category': test_case['category'],
                'expected_anomalous': test_case['expected_anomalous'],
                'predicted_anomalous': is_anomalous,
                'confidence': confidence,
                'message': message,
                'success': True,
                'correct': correct,
                'simulation_time': data.get('simulation_time', 0)
            }
            
        except Exception as e:
            return {
                'name': test_name,
                'category': test_case['category'],
                'expected_anomalous': test_case['expected_anomalous'],
                'success': False,
                'stage_failed': 'exception',
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all test cases"""
        self.log("\n" + "=" * 80)
        self.log("STARTING COMPREHENSIVE TEST SUITE", "TEST")
        self.log("=" * 80)
        
        total = len(self.test_cases)
        
        for i, test_case in enumerate(self.test_cases, 1):
            self.log(f"\n[{i:2d}/{total:2d}] Testing: {test_case['name']}", "TEST")
            self.log(f"         Category: {test_case['category']}")
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            if result['success']:
                if result['correct']:
                    status = "SUCCESS"
                    expected_str = "ANOMALOUS" if result['expected_anomalous'] else "NORMAL"
                    predicted_str = "ANOMALOUS" if result['predicted_anomalous'] else "NORMAL"
                    self.log(f"         ✅ CORRECT | Expected: {expected_str}, Got: {predicted_str}", status)
                    self.log(f"         Confidence: {result['confidence']:.3f}")
                else:
                    status = "ERROR"
                    expected_str = "ANOMALOUS" if result['expected_anomalous'] else "NORMAL"
                    predicted_str = "ANOMALOUS" if result['predicted_anomalous'] else "NORMAL"
                    self.log(f"         ❌ WRONG | Expected: {expected_str}, Got: {predicted_str}", status)
                    self.log(f"         Confidence: {result['confidence']:.3f}")
            else:
                self.log(f"         ❌ FAILED at {result.get('stage_failed', 'unknown')}", "ERROR")
                self.log(f"         Error: {result.get('error', 'Unknown')}")
            
            # Small delay between tests
            time.sleep(3)
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics"""
        if not self.results:
            self.log("No results to analyze", "ERROR")
            return
        
        self.log("\n" + "=" * 80)
        self.log("TEST RESULTS SUMMARY", "TEST")
        self.log("=" * 80)
        
        # Overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - successful_tests
        
        # For successful tests only
        completed_results = [r for r in self.results if r['success']]
        
        if completed_results:
            correct_predictions = sum(1 for r in completed_results if r['correct'])
            accuracy = correct_predictions / len(completed_results)
            
            # Extract for confusion matrix
            y_true = [r['expected_anomalous'] for r in completed_results]
            y_pred = [r['predicted_anomalous'] for r in completed_results]
            
            if SKLEARN_AVAILABLE:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
            else:
                # Manual calculation
                tp = sum(1 for i in range(len(y_true)) if y_true[i] and y_pred[i])
                tn = sum(1 for i in range(len(y_true)) if not y_true[i] and not y_pred[i])
                fp = sum(1 for i in range(len(y_true)) if not y_true[i] and y_pred[i])
                fn = sum(1 for i in range(len(y_true)) if y_true[i] and not y_pred[i])
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                cm = [[tn, fp], [fn, tp]]
            
            self.log(f"\nPipeline Completion:")
            self.log(f"  Total Tests:      {total_tests}")
            self.log(f"  Completed:        {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
            self.log(f"  Failed:           {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
            
            self.log(f"\nVAE Performance (on completed tests):")
            self.log(f"  Accuracy:         {accuracy:.3f} ({correct_predictions}/{len(completed_results)})")
            self.log(f"  Precision:        {precision:.3f}")
            self.log(f"  Recall:           {recall:.3f}")
            self.log(f"  F1-Score:         {f1:.3f}")
            
            self.log(f"\nConfusion Matrix:")
            self.log(f"                  Predicted")
            self.log(f"                  Normal  Anomalous")
            self.log(f"  Actual Normal     {cm[0][0]:3d}      {cm[0][1]:3d}")
            self.log(f"      Anomalous     {cm[1][0]:3d}      {cm[1][1]:3d}")
            
            # Category-wise performance
            self.log(f"\nCategory-wise Performance:")
            categories = {}
            for r in completed_results:
                cat = r['category']
                if cat not in categories:
                    categories[cat] = {'correct': 0, 'total': 0, 'confidences': []}
                categories[cat]['correct'] += r['correct']
                categories[cat]['total'] += 1
                categories[cat]['confidences'].append(r['confidence'])
            
            for cat, stats in sorted(categories.items()):
                cat_accuracy = stats['correct'] / stats['total']
                mean_conf = np.mean(stats['confidences'])
                self.log(f"  {cat:30s}: {cat_accuracy:.3f} ({stats['correct']:2d}/{stats['total']:2d}) | Conf: {mean_conf:.3f}")
            
            # Average simulation time
            avg_sim_time = np.mean([r['simulation_time'] for r in completed_results if 'simulation_time' in r])
            self.log(f"\nAverage Simulation Time: {avg_sim_time:.3f}s")
        
        # Failure analysis
        if failed_tests > 0:
            self.log(f"\nFailure Analysis:")
            failure_stages = {}
            for r in self.results:
                if not r['success']:
                    stage = r.get('stage_failed', 'unknown')
                    failure_stages[stage] = failure_stages.get(stage, 0) + 1
            
            for stage, count in sorted(failure_stages.items()):
                self.log(f"  {stage:20s}: {count} failures")
        
        # Total time
        total_time = time.time() - self.start_time
        self.log(f"\nTotal Test Time: {total_time:.2f}s")
        self.log(f"Average per test: {total_time/total_tests:.2f}s")
        
        # Final verdict
        if successful_tests == total_tests and correct_predictions == len(completed_results):
            self.log("\n🎉 ALL TESTS PASSED!", "SUCCESS")
        elif successful_tests > total_tests * 0.8:
            self.log(f"\n✅ STRONG PERFORMANCE: {successful_tests/total_tests*100:.1f}% completion rate", "SUCCESS")
        else:
            self.log(f"\n⚠️  MODERATE PERFORMANCE: {successful_tests/total_tests*100:.1f}% completion rate", "WARNING")
        
        self.log("=" * 80)


def main():
    """Main test execution"""
    print("\n🧪 Comprehensive API Integration Test Suite\n")
    
    # Initialize test suite
    test_suite = ComprehensiveAPITestSuite()
    
    # Step 1: Health check
    if not test_suite.check_server_health():
        print("\n❌ Server health check failed - aborting tests")
        sys.exit(1)
    
    # Step 2: Create test dataset
    test_suite.create_test_dataset()
    
    # Step 3: Run all tests
    test_suite.run_all_tests()
    
    # Step 4: Calculate and display metrics
    test_suite.calculate_metrics()
    
    print("\n✅ Test suite completed!\n")


if __name__ == "__main__":
    main()