# test_api_endpoints.py
import requests
import json
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 60

def print_test(name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

def test_health():
    """Test health endpoint"""
    print_test("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_system_status():
    """Test system status endpoint"""
    print_test("System Status")
    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=10)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_available_models():
    """Test get available models"""
    print_test("Available Models")
    try:
        response = requests.get(f"{BASE_URL}/api/models/available", timeout=10)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total models: {data.get('total_count', 0)}")
        for model in data.get('models', []):
            print(f"  - {model['id']}: {model['name']} ({model['provider']})")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_model_recommendations():
    """Test model recommendations"""
    print_test("Model Recommendations")
    try:
        task_types = ["verilog_generation", "analysis", "mermaid_generation"]
        for task in task_types:
            response = requests.get(f"{BASE_URL}/api/models/recommendations/{task}", timeout=10)
            data = response.json()
            print(f"\n{task}:")
            for rec in data.get('recommendations', []):
                print(f"  - {rec['id']}: {rec['name']}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_model_test():
    """Test model testing endpoint"""
    print_test("Model Test")
    try:
        models_to_test = ["groq", "claude", "gemini/gemini-2.0-flash-exp"]
        for model in models_to_test:
            print(f"\nTesting {model}...")
            response = requests.post(
                f"{BASE_URL}/api/models/test",
                json={"model": model},
                timeout=30
            )
            data = response.json()
            if data.get('success'):
                print(f"  ✓ {model} working")
            else:
                print(f"  ✗ {model} failed: {data.get('error', 'Unknown error')}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_generate_mermaid():
    """Test Mermaid generation"""
    print_test("Generate Mermaid Diagram")
    try:
        payload = {
            "prompt": "Create a simple 2-bit counter with clock and reset",
            "model": "claude"
        }
        response = requests.post(
            f"{BASE_URL}/api/design/generate-mermaid",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if data.get('success'):
            print(f"Model used: {data.get('model_used')}")
            print(f"Session ID: {data.get('session_id')}")
            print(f"\nMermaid code preview:")
            print(data.get('mermaid_code', '')[:200])
            return True, data
        else:
            print(f"Failed: {data.get('error')}")
            return False, None
    except Exception as e:
        print(f"Failed: {e}")
        return False, None

def test_generate_verilog(mermaid_code=None):
    """Test Verilog generation"""
    print_test("Generate Verilog Code")
    try:
        if not mermaid_code:
            mermaid_code = """graph TD
    CLK[Clock] --> COUNTER[2-bit Counter]
    RST[Reset] --> COUNTER
    COUNTER --> Q[Count Output]"""
        
        payload = {
            "mermaid_code": mermaid_code,
            "description": "Simple 2-bit counter",
            "model": "claude",
            "use_rag": False
        }
        response = requests.post(
            f"{BASE_URL}/api/design/generate-verilog",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if data.get('success'):
            print(f"Model used: {data.get('model_used')}")
            print(f"Stats: {data.get('stats')}")
            print(f"Validation issues: {len(data.get('validation_issues', []))}")
            print(f"\nVerilog code preview:")
            print(data.get('verilog_code', '')[:300])
            return True, data
        else:
            print(f"Failed: {data.get('error')}")
            return False, None
    except Exception as e:
        print(f"Failed: {e}")
        return False, None

def test_simulation(verilog_code=None):
    """Test simulation"""
    print_test("Run Simulation")
    try:
        if not verilog_code:
            # Simple testable Verilog code
            verilog_code = """`timescale 1ns/1ps

module counter(
    input wire clk,
    input wire rst_n,
    output reg [1:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 2'b00;
    else
        count <= count + 1'b1;
end

endmodule

module testbench;
    reg clk, rst_n;
    wire [1:0] count;
    
    counter dut (
        .clk(clk),
        .rst_n(rst_n),
        .count(count)
    );
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, testbench);
        
        clk = 0;
        rst_n = 0;
        #20 rst_n = 1;
        #100 $finish;
    end
    
    always #5 clk = ~clk;
    
    initial begin
        $monitor("Time=%0t clk=%b rst_n=%b count=%d", 
                 $time, clk, rst_n, count);
    end
endmodule"""
        
        payload = {"verilog_code": verilog_code}
        response = requests.post(
            f"{BASE_URL}/api/simulation/run",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if data.get('success'):
            print(f"Simulation time: {data.get('simulation_time', 0):.2f}s")
            print(f"Has waveform: {'Yes' if data.get('waveform_csv') else 'No'}")
            if data.get('error'):
                print(f"Warning: {data.get('error')}")
            return True, data
        else:
            print(f"Failed: {data.get('error')}")
            return False, None
    except Exception as e:
        print(f"Failed: {e}")
        return False, None

def test_analysis(waveform_csv=None, verilog_code=None):
    """Test waveform analysis"""
    print_test("Analyze Waveform")
    try:
        if not waveform_csv:
            waveform_csv = """timestamp,signal,value
0,clk,0
0,rst_n,0
0,count,0
5,clk,1
10,clk,0
15,clk,1
20,clk,0
20,rst_n,1"""
        
        payload = {
            "waveform_csv": waveform_csv,
            "verilog_code": verilog_code or "",
            "model": "claude"
        }
        response = requests.post(
            f"{BASE_URL}/api/analysis/analyze-waveform",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if data.get('success'):
            print(f"Model used: {data.get('model_used')}")
            print(f"\nAnalysis preview:")
            print(data.get('analysis_results', '')[:300])
            return True, data
        else:
            print(f"Failed: {data.get('error')}")
            return False, None
    except Exception as e:
        print(f"Failed: {e}")
        return False, None

def test_full_pipeline():
    """Test complete pipeline from prompt to analysis"""
    print_test("FULL PIPELINE TEST")
    
    # Step 1: Generate Mermaid
    print("\nStep 1: Generating Mermaid diagram...")
    success, mermaid_data = test_generate_mermaid()
    if not success:
        print("Pipeline failed at Mermaid generation")
        return False
    
    # Step 2: Generate Verilog
    print("\nStep 2: Generating Verilog code...")
    success, verilog_data = test_generate_verilog(mermaid_data['mermaid_code'])
    if not success:
        print("Pipeline failed at Verilog generation")
        return False
    
    # Step 3: Run Simulation
    print("\nStep 3: Running simulation...")
    success, sim_data = test_simulation(verilog_data['verilog_code'])
    if not success:
        print("Pipeline failed at simulation")
        return False
    
    # Step 4: Analyze Waveform (if available)
    if sim_data.get('waveform_csv'):
        print("\nStep 4: Analyzing waveform...")
        success, analysis_data = test_analysis(
            sim_data['waveform_csv'],
            verilog_data['verilog_code']
        )
        if not success:
            print("Pipeline failed at analysis")
            return False
    
    print("\n✓ Full pipeline completed successfully!")
    return True

if __name__ == "__main__":
    print("Testing FastAPI Endpoints")
    print(f"Base URL: {BASE_URL}\n")
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/api/health", timeout=5)
    except:
        print(f"Error: Server not running at {BASE_URL}")
        print("Start the server with: python main.py")
        sys.exit(1)
    
    results = {}
    
    # Basic tests
    results['health'] = test_health()
    results['system_status'] = test_system_status()
    results['available_models'] = test_available_models()
    results['model_recommendations'] = test_model_recommendations()
    results['model_test'] = test_model_test()
    
    # Individual endpoint tests
    results['generate_mermaid'] = test_generate_mermaid()[0]
    results['generate_verilog'] = test_generate_verilog()[0]
    results['simulation'] = test_simulation()[0]
    results['analysis'] = test_analysis()[0]
    
    # Full pipeline test
    results['full_pipeline'] = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    sys.exit(0 if passed == total else 1)