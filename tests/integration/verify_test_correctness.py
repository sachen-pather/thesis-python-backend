"""
Ground Truth Verification Suite - FIXED FOR LONG FORMAT CSV
Validates that test circuits produce CORRECT functional behavior
This helps determine if VAE predictions are accurate or if there are real bugs

Run from project root:
    python tests/integration/verify_test_correctness.py
"""

import sys
import os
import requests
import pandas as pd
from io import StringIO

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

BASE_URL = "http://localhost:8000"


class GroundTruthVerifier:
    """Verify circuit functionality against known-good behavior"""
    
    def __init__(self):
        self.results = []
    
    def log(self, message, status="INFO"):
        emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}.get(status, "📝")
        print(f"{emoji} {message}")
    
    def parse_waveform(self, csv_data):
        """Parse waveform CSV (handles long format) into wide DataFrame"""
        try:
            df_long = pd.read_csv(StringIO(csv_data))
            
            # Check if it's long format (has 'signal' and 'value' columns)
            if 'signal' in df_long.columns and 'value' in df_long.columns:
                # Pivot from long to wide format
                df_wide = df_long.pivot(index='timestamp', columns='signal', values='value')
                df_wide = df_wide.reset_index()
                return df_wide
            else:
                # Already in wide format
                return df_long
                
        except Exception as e:
            self.log(f"Failed to parse waveform: {e}", "ERROR")
            return None
    
    def verify_and_gate(self, waveform_df):
        """Verify 2-input AND gate truth table"""
        try:
            # Look for input columns (a, b, x, y, in1, in2, etc.)
            cols = [c.lower() for c in waveform_df.columns]
            
            # Find inputs
            a_col = None
            b_col = None
            out_col = None
            
            for col in waveform_df.columns:
                col_lower = col.lower()
                if col_lower in ['a', 'x', 'in1', 'input1', 'i0'] and a_col is None:
                    a_col = col
                elif col_lower in ['b', 'y', 'in2', 'input2', 'i1'] and b_col is None:
                    b_col = col
                elif col_lower in ['out', 'z', 'output', 'result', 'o', 'q'] and out_col is None:
                    out_col = col
            
            if not a_col or not b_col or not out_col:
                return False, f"Missing signals (a={a_col}, b={b_col}, out={out_col})"
            
            self.log(f"  Found signals: a='{a_col}', b='{b_col}', out='{out_col}'")
            
            # Skip initial rows (reset/initialization)
            df_stable = waveform_df.iloc[20:].copy()
            
            # Check AND gate truth table
            correct = 0
            total = 0
            errors = []
            
            for idx, row in df_stable.iterrows():
                a = int(row[a_col]) if pd.notna(row[a_col]) else 0
                b = int(row[b_col]) if pd.notna(row[b_col]) else 0
                out = int(row[out_col]) if pd.notna(row[out_col]) else 0
                
                expected = a & b
                
                if out == expected:
                    correct += 1
                else:
                    if len(errors) < 3:  # Store first 3 errors
                        errors.append(f"a={a}, b={b} → out={out} (expected {expected})")
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            if accuracy >= 0.9:
                return True, f"AND gate correct: {accuracy*100:.1f}% ({correct}/{total})"
            else:
                error_str = "; ".join(errors) if errors else "various errors"
                return False, f"AND gate faulty: {accuracy*100:.1f}% ({correct}/{total}). Errors: {error_str}"
                
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def verify_or_gate(self, waveform_df):
        """Verify 2-input OR gate truth table"""
        try:
            cols = [c.lower() for c in waveform_df.columns]
            
            a_col = None
            b_col = None
            out_col = None
            
            for col in waveform_df.columns:
                col_lower = col.lower()
                if col_lower in ['a', 'x', 'in1', 'input1', 'i0'] and a_col is None:
                    a_col = col
                elif col_lower in ['b', 'y', 'in2', 'input2', 'i1'] and b_col is None:
                    b_col = col
                elif col_lower in ['out', 'z', 'output', 'result', 'o', 'q'] and out_col is None:
                    out_col = col
            
            if not a_col or not b_col or not out_col:
                return False, f"Missing signals (a={a_col}, b={b_col}, out={out_col})"
            
            self.log(f"  Found signals: a='{a_col}', b='{b_col}', out='{out_col}'")
            
            df_stable = waveform_df.iloc[20:].copy()
            
            correct = 0
            total = 0
            errors = []
            
            for idx, row in df_stable.iterrows():
                a = int(row[a_col]) if pd.notna(row[a_col]) else 0
                b = int(row[b_col]) if pd.notna(row[b_col]) else 0
                out = int(row[out_col]) if pd.notna(row[out_col]) else 0
                
                expected = a | b
                
                if out == expected:
                    correct += 1
                else:
                    if len(errors) < 3:
                        errors.append(f"a={a}, b={b} → out={out} (expected {expected})")
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            if accuracy >= 0.9:
                return True, f"OR gate correct: {accuracy*100:.1f}% ({correct}/{total})"
            else:
                error_str = "; ".join(errors) if errors else "various errors"
                return False, f"OR gate faulty: {accuracy*100:.1f}% ({correct}/{total}). Errors: {error_str}"
                
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def verify_xor_gate(self, waveform_df):
        """Verify 2-input XOR gate truth table"""
        try:
            a_col = None
            b_col = None
            out_col = None
            
            for col in waveform_df.columns:
                col_lower = col.lower()
                if col_lower in ['a', 'x', 'in1', 'input1', 'i0'] and a_col is None:
                    a_col = col
                elif col_lower in ['b', 'y', 'in2', 'input2', 'i1'] and b_col is None:
                    b_col = col
                elif col_lower in ['out', 'z', 'output', 'result', 'o', 'q'] and out_col is None:
                    out_col = col
            
            if not a_col or not b_col or not out_col:
                return False, f"Missing signals (a={a_col}, b={b_col}, out={out_col})"
            
            self.log(f"  Found signals: a='{a_col}', b='{b_col}', out='{out_col}'")
            
            df_stable = waveform_df.iloc[20:].copy()
            
            correct = 0
            total = 0
            errors = []
            
            for idx, row in df_stable.iterrows():
                a = int(row[a_col]) if pd.notna(row[a_col]) else 0
                b = int(row[b_col]) if pd.notna(row[b_col]) else 0
                out = int(row[out_col]) if pd.notna(row[out_col]) else 0
                
                expected = a ^ b
                
                if out == expected:
                    correct += 1
                else:
                    if len(errors) < 3:
                        errors.append(f"a={a}, b={b} → out={out} (expected {expected})")
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            if accuracy >= 0.9:
                return True, f"XOR gate correct: {accuracy*100:.1f}% ({correct}/{total})"
            else:
                error_str = "; ".join(errors) if errors else "various errors"
                return False, f"XOR gate faulty: {accuracy*100:.1f}% ({correct}/{total}). Errors: {error_str}"
                
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def verify_counter(self, waveform_df):
        """Verify counter increments correctly"""
        try:
            # Find counter output
            count_col = None
            for col in waveform_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['count', 'cnt', 'q', 'out']):
                    # Check if it's multi-bit (varies more than just 0/1)
                    unique_vals = waveform_df[col].nunique()
                    if unique_vals > 2:
                        count_col = col
                        break
            
            if not count_col:
                return False, "Could not identify counter output (needs > 2 unique values)"
            
            self.log(f"  Found counter: '{count_col}'")
            
            df_stable = waveform_df.iloc[20:].copy()
            
            # Check if counter increments
            values = df_stable[count_col].dropna().tolist()
            
            if len(values) < 3:
                return False, "Not enough data points"
            
            # Count increments
            increments = 0
            for i in range(1, len(values)):
                if values[i] == (values[i-1] + 1) % 16:  # 4-bit wraps at 16
                    increments += 1
                elif values[i] == values[i-1]:  # Stable is ok
                    pass
            
            increment_rate = increments / (len(values) - 1) if len(values) > 1 else 0
            
            if increment_rate >= 0.2:  # At least 20% incrementing
                return True, f"Counter working: {increment_rate*100:.1f}% increment rate"
            else:
                return False, f"Counter not incrementing: {increment_rate*100:.1f}% increment rate"
                
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def test_circuit(self, prompt, circuit_type, verifier_func):
        """Test a single circuit with ground truth verification"""
        self.log(f"\nTesting: {circuit_type}")
        
        try:
            # Step 1: Generate Mermaid
            response = requests.post(
                f"{BASE_URL}/api/design/generate-mermaid",
                json={"prompt": prompt, "model": "groq"},
                timeout=30
            )
            
            if response.status_code != 200 or not response.json().get("success"):
                self.log(f"  Mermaid generation failed", "ERROR")
                return None
            
            mermaid_code = response.json()['mermaid_code']
            
            # Step 2: Generate Verilog
            response = requests.post(
                f"{BASE_URL}/api/design/generate-verilog",
                json={"mermaid_code": mermaid_code, "description": prompt, "model": "groq"},
                timeout=30
            )
            
            if response.status_code != 200 or not response.json().get("success"):
                self.log(f"  Verilog generation failed", "ERROR")
                return None
            
            verilog_code = response.json()['verilog_code']
            
            # Step 3: Simulate with VAE
            response = requests.post(
                f"{BASE_URL}/api/simulation/run-with-verification",
                json={"verilog_code": verilog_code},
                timeout=60
            )
            
            if response.status_code != 200:
                self.log(f"  Simulation failed", "ERROR")
                return None
            
            data = response.json()
            
            if not data.get("success"):
                self.log(f"  Simulation failed", "ERROR")
                return None
            
            # Get VAE prediction
            vae = data.get("verification", {}).get("vae_verification", {})
            vae_prediction = vae.get("is_anomalous", False)
            vae_confidence = vae.get("confidence", 0.0)
            
            # Get waveform
            csv_data = data.get("waveform_csv", "")
            if not csv_data:
                self.log(f"  No waveform data", "ERROR")
                return None
            
            # Parse waveform (handles long format now)
            waveform_df = self.parse_waveform(csv_data)
            if waveform_df is None:
                return None
            
            self.log(f"  Waveform parsed: {len(waveform_df)} rows, {len(waveform_df.columns)} signals")
            
            # Ground truth verification
            is_correct, message = verifier_func(waveform_df)
            
            # Compare VAE vs Ground Truth
            vae_str = "🔴 ANOMALOUS" if vae_prediction else "✅ NORMAL"
            ground_truth_str = "✅ WORKS" if is_correct else "❌ BROKEN"
            
            self.log(f"  Ground Truth:  {ground_truth_str}")
            self.log(f"  Details:       {message}")
            self.log(f"  VAE Predicted: {vae_str} (confidence: {vae_confidence:.3f})")
            
            # VAE should predict ANOMALOUS if circuit is broken, NORMAL if working
            vae_correct = (vae_prediction == (not is_correct))
            
            if vae_correct:
                self.log(f"  VAE Verdict:   ✅ CORRECT", "SUCCESS")
            else:
                self.log(f"  VAE Verdict:   ❌ WRONG", "ERROR")
                if is_correct and vae_prediction:
                    self.log(f"                 (False Positive: marked good circuit as anomalous)", "WARNING")
                elif not is_correct and not vae_prediction:
                    self.log(f"                 (False Negative: missed a bug)", "WARNING")
            
            result = {
                'circuit_type': circuit_type,
                'ground_truth_correct': is_correct,
                'vae_anomalous': vae_prediction,
                'vae_confidence': vae_confidence,
                'vae_correct': vae_correct,
                'message': message
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.log(f"  Exception: {e}", "ERROR")
            return None
    
    def run_verification_suite(self):
        """Run ground truth verification on key circuits"""
        self.log("=" * 80)
        self.log("GROUND TRUTH VERIFICATION SUITE (FIXED)")
        self.log("=" * 80)
        
        # Test logic gates
        self.test_circuit("Create a 2-input AND gate", "AND Gate", self.verify_and_gate)
        self.test_circuit("Design a 2-input OR gate", "OR Gate", self.verify_or_gate)
        self.test_circuit("Build a 2-input XOR gate", "XOR Gate", self.verify_xor_gate)
        
        # Test counter
        self.test_circuit("Create a 4-bit binary up counter with clock and reset", "4-bit Counter", self.verify_counter)
    
    def print_summary(self):
        """Print verification summary"""
        self.log("\n" + "=" * 80)
        self.log("VERIFICATION SUMMARY")
        self.log("=" * 80)
        
        if not self.results:
            self.log("No results to analyze", "ERROR")
            return
        
        total = len(self.results)
        ground_truth_correct = sum(1 for r in self.results if r['ground_truth_correct'])
        vae_correct = sum(1 for r in self.results if r['vae_correct'])
        
        false_positives = sum(1 for r in self.results if r['ground_truth_correct'] and r['vae_anomalous'])
        false_negatives = sum(1 for r in self.results if not r['ground_truth_correct'] and not r['vae_anomalous'])
        
        self.log(f"\nTotal Circuits Tested: {total}")
        self.log(f"Functionally Correct:  {ground_truth_correct}/{total} ({ground_truth_correct/total*100:.1f}%)")
        self.log(f"VAE Correct:           {vae_correct}/{total} ({vae_correct/total*100:.1f}%)")
        self.log(f"False Positives:       {false_positives} (good circuits marked as anomalous)")
        self.log(f"False Negatives:       {false_negatives} (buggy circuits marked as normal)")
        
        self.log("\nDetailed Results:")
        for r in self.results:
            gt_str = "✅ WORKS" if r['ground_truth_correct'] else "❌ BROKEN"
            vae_str = "🔴 ANOMALOUS" if r['vae_anomalous'] else "✅ NORMAL"
            vae_verdict = "✅ CORRECT" if r['vae_correct'] else "❌ WRONG"
            
            self.log(f"\n  {r['circuit_type']}")
            self.log(f"    Functional: {gt_str}")
            self.log(f"    VAE:        {vae_str} → {vae_verdict}")
            self.log(f"    Details:    {r['message']}")
        
        # Key insight
        self.log("\n" + "=" * 80)
        if ground_truth_correct == total and vae_correct < total:
            self.log("⚠️  INSIGHT: All circuits are FUNCTIONALLY CORRECT", "WARNING")
            self.log("⚠️  But VAE is marking some as ANOMALOUS (False Positives)", "WARNING")
            self.log("⚠️  This suggests the VAE threshold may be too sensitive!", "WARNING")
        elif ground_truth_correct < total and vae_correct == total:
            self.log("✅ INSIGHT: VAE correctly identified all bugs!", "SUCCESS")
        elif ground_truth_correct == total and vae_correct == total:
            self.log("✅ INSIGHT: All circuits work AND VAE is 100% accurate!", "SUCCESS")
        else:
            self.log("ℹ️  INSIGHT: Mixed results - some bugs detected, some false positives", "INFO")
        
        self.log("=" * 80)


def main():
    verifier = GroundTruthVerifier()
    verifier.run_verification_suite()
    verifier.print_summary()


if __name__ == "__main__":
    main()