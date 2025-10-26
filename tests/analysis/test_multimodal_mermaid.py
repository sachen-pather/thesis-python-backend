# tests/analysis/test_multimodal_mermaid.py
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import requests


class MultimodalCircuitLoader:
    """Load circuit prompts, mermaid diagrams, and combined files"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.prompts_file = self.root_dir / "multimodal" / "prompts" / "all_prompts.json"
        self.mermaid_dir = self.root_dir / "multimodal" / "mermaid" / "individual_files"
        
        print(f"📁 Root: {self.root_dir}")
        print(f"📄 Prompts: {self.prompts_file.name}")
        print(f"🎨 Mermaid: {self.mermaid_dir.name}")
    
    def load_prompts(self):
        """Load all circuit prompts from JSON"""
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")
        
        with open(self.prompts_file, 'r') as f:
            data = json.load(f)
        
        if 'prompts' in data:
            circuits = data['prompts']
        elif 'circuits' in data:
            circuits = data['circuits']
        elif isinstance(data, list):
            circuits = data
        else:
            raise ValueError(f"Unexpected JSON structure. Keys found: {list(data.keys())}")
        
        print(f"✓ Loaded {len(circuits)} circuits")
        return circuits
    
    def normalize_filename(self, circuit_name):
        """Convert circuit name to filename format"""
        filename = circuit_name.lower()
        filename = filename.replace('-', '_')
        filename = filename.replace(':', '_to_')
        filename = filename.replace(' ', '_')
        return filename
    
    def load_mermaid(self, circuit_name):
        """Load Mermaid diagram for a specific circuit"""
        filename = self.normalize_filename(circuit_name) + '.mmd'
        mermaid_path = self.mermaid_dir / filename
        
        if not mermaid_path.exists():
            return None
        
        with open(mermaid_path, 'r') as f:
            return f.read()
    
    def get_circuit_data(self, circuit_id):
        """Get all data for a specific circuit"""
        circuits = self.load_prompts()
        circuit = next((c for c in circuits if c['id'] == circuit_id), None)
        
        if not circuit:
            raise ValueError(f"Circuit with id {circuit_id} not found")
        
        mermaid_content = self.load_mermaid(circuit['name'])
        
        return {
            'id': circuit['id'],
            'name': circuit['name'],
            'prompt': circuit['prompt'],
            'category': circuit['category'],
            'complexity': circuit['complexity'],
            'mermaid': mermaid_content,
            'has_mermaid': mermaid_content is not None
        }
    
    def get_all_circuits(self):
        """Get all circuit data"""
        circuits = self.load_prompts()
        return [self.get_circuit_data(c['id']) for c in circuits]


class ThreeWayTester:
    """Test circuits with three approaches via backend API"""
    
    def __init__(self, backend_url="http://localhost:8000", model="claude"):
        self.backend_url = backend_url.rstrip('/')
        self.model = model
        
        print(f"\n🔧 Backend: {self.backend_url}")
        print(f"   Model: {model}")
        
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to backend"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ Connected")
                return
        except:
            pass
        
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            print(f"   ✅ Connected")
            return
        except Exception as e:
            raise RuntimeError(f"❌ Cannot connect to backend at {self.backend_url}")
    
    def generate_verilog(self, prompt, mermaid_diagram, approach_name):
        """Generate Verilog via backend API"""
        print(f"   🤖 Generating ({approach_name})...", end=" ")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.backend_url}/api/design/generate-verilog",
                json={
                    "mermaid_code": mermaid_diagram,
                    "description": prompt,
                    "model": self.model,
                    "use_rag": False,
                    "session_id": None
                },
                timeout=180
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code != 200:
                print(f"✗ API error ({response.status_code})")
                return None
            
            data = response.json()
            
            if not data.get('success', False):
                print(f"✗ {data.get('error', 'Unknown error')}")
                return None
            
            verilog_code = data.get('verilog_code')
            if not verilog_code:
                print(f"✗ No code returned")
                return None
            
            lines = len(verilog_code.split('\n'))
            print(f"✓ {lines} lines ({generation_time:.1f}s)")
            
            return {
                'code': verilog_code,
                'generation_time': generation_time,
                'lines_of_code': lines,
                'has_testbench': 'testbench' in verilog_code.lower()
            }
            
        except requests.exceptions.Timeout:
            print(f"✗ Timeout")
            return None
        except Exception as e:
            print(f"✗ {str(e)[:50]}")
            return None
    
    def simulate_and_verify(self, verilog_code, approach_name):
        """Simulate and verify via backend API"""
        print(f"   🔨 Simulating ({approach_name})...", end=" ")
        
        sim_start = time.time()
        
        try:
            response = requests.post(
                f"{self.backend_url}/api/simulation/run-with-verification",
                json={
                    "verilog_code": verilog_code,
                    "model": self.model,
                    "session_id": None
                },
                timeout=300
            )
            
            sim_time = time.time() - sim_start
            
            if response.status_code != 200:
                print(f"✗ API error ({response.status_code})")
                return None, sim_time
            
            data = response.json()
            success = data.get('success', False)
            csv_data = data.get('waveform_csv', '')
            
            # Get verification results
            verification = data.get('verification', {})
            vae_verif = verification.get('vae_verification', {})
            llm_verif = verification.get('llm_verification', {})
            
            # Determine status
            status_parts = []
            if success:
                status_parts.append("✓")
            else:
                status_parts.append("✗")
                
            if csv_data and len(csv_data) > 50:
                status_parts.append("Wave")
            
            # LLM verdict
            if llm_verif.get('available'):
                analysis = llm_verif.get('analysis', '')
                if 'VERDICT: ANOMALOUS' in analysis:
                    status_parts.append("LLM:ANOM")
                else:
                    status_parts.append("LLM:OK")
            
            # VAE verdict
            if vae_verif.get('available'):
                if vae_verif.get('is_anomalous'):
                    status_parts.append("VAE:ANOM")
                else:
                    status_parts.append("VAE:OK")
            
            print(f"{' '.join(status_parts)} ({sim_time:.1f}s)")
            
            return data, sim_time
            
        except requests.exceptions.Timeout:
            print(f"✗ Timeout")
            return None, time.time() - sim_start
        except Exception as e:
            print(f"✗ {str(e)[:50]}")
            return None, time.time() - sim_start
    
    def test_three_ways(self, circuit):
        """Test a circuit with all three approaches"""
        print(f"\n{'='*70}")
        print(f"🧪 {circuit['id']}. {circuit['name']} ({circuit['complexity']}, {circuit['category']})")
        print(f"{'='*70}")
        
        results = {
            'circuit_id': circuit['id'],
            'circuit_name': circuit['name'],
            'category': circuit['category'],
            'complexity': circuit['complexity'],
            'original_prompt': circuit['prompt'],
            'model': self.model
        }
        
        # Approach 1: Prompt Only
        print(f"[1/3] PROMPT ONLY")
        prompt_result = self.generate_verilog(
            prompt=circuit['prompt'],
            mermaid_diagram="",
            approach_name="prompt"
        )
        
        if prompt_result:
            sim_data, sim_time = self.simulate_and_verify(prompt_result['code'], "prompt")
            
            if sim_data:
                llm_analysis = sim_data.get('verification', {}).get('llm_verification', {}).get('analysis', '')
                results['prompt_only'] = {
                    'success': sim_data.get('success', False),
                    'compiled': sim_data.get('success', False),
                    'simulated': sim_data.get('success', False),
                    'has_waveform': bool(sim_data.get('waveform_csv') and len(sim_data.get('waveform_csv', '')) > 50),
                    'generation_time': prompt_result['generation_time'],
                    'simulation_time': sim_time,
                    'lines_of_code': prompt_result['lines_of_code'],
                    'anomalous': 'VERDICT: ANOMALOUS' in llm_analysis if llm_analysis else None,
                    'vae_anomalous': sim_data.get('verification', {}).get('vae_verification', {}).get('is_anomalous'),
                }
            else:
                results['prompt_only'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
        else:
            results['prompt_only'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
        
        # Approach 2: Mermaid Only
        print(f"[2/3] MERMAID ONLY")
        if circuit['has_mermaid']:
            mermaid_result = self.generate_verilog(
                prompt="",
                mermaid_diagram=circuit['mermaid'],
                approach_name="mermaid"
            )
            
            if mermaid_result:
                sim_data, sim_time = self.simulate_and_verify(mermaid_result['code'], "mermaid")
                
                if sim_data:
                    llm_analysis = sim_data.get('verification', {}).get('llm_verification', {}).get('analysis', '')
                    results['mermaid_only'] = {
                        'success': sim_data.get('success', False),
                        'compiled': sim_data.get('success', False),
                        'simulated': sim_data.get('success', False),
                        'has_waveform': bool(sim_data.get('waveform_csv') and len(sim_data.get('waveform_csv', '')) > 50),
                        'generation_time': mermaid_result['generation_time'],
                        'simulation_time': sim_time,
                        'lines_of_code': mermaid_result['lines_of_code'],
                        'anomalous': 'VERDICT: ANOMALOUS' in llm_analysis if llm_analysis else None,
                        'vae_anomalous': sim_data.get('verification', {}).get('vae_verification', {}).get('is_anomalous'),
                    }
                else:
                    results['mermaid_only'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
            else:
                results['mermaid_only'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
        else:
            print(f"   ⚠️  No Mermaid - SKIP")
            results['mermaid_only'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
        
        # Approach 3: Combined
        print(f"[3/3] COMBINED")
        if circuit['has_mermaid']:
            combined_result = self.generate_verilog(
                prompt=circuit['prompt'],
                mermaid_diagram=circuit['mermaid'],
                approach_name="combined"
            )
            
            if combined_result:
                sim_data, sim_time = self.simulate_and_verify(combined_result['code'], "combined")
                
                if sim_data:
                    llm_analysis = sim_data.get('verification', {}).get('llm_verification', {}).get('analysis', '')
                    results['combined'] = {
                        'success': sim_data.get('success', False),
                        'compiled': sim_data.get('success', False),
                        'simulated': sim_data.get('success', False),
                        'has_waveform': bool(sim_data.get('waveform_csv') and len(sim_data.get('waveform_csv', '')) > 50),
                        'generation_time': combined_result['generation_time'],
                        'simulation_time': sim_time,
                        'lines_of_code': combined_result['lines_of_code'],
                        'anomalous': 'VERDICT: ANOMALOUS' in llm_analysis if llm_analysis else None,
                        'vae_anomalous': sim_data.get('verification', {}).get('vae_verification', {}).get('is_anomalous'),
                    }
                else:
                    results['combined'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
            else:
                results['combined'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
        else:
            print(f"   ⚠️  No Mermaid - SKIP")
            results['combined'] = {'success': False, 'compiled': False, 'simulated': False, 'has_waveform': False, 'anomalous': None}
        
        # Summary
        print(f"\n📊 Summary:")
        for approach in ['prompt_only', 'mermaid_only', 'combined']:
            data = results[approach]
            status = "✅" if (data.get('compiled') and data.get('simulated') and data.get('has_waveform') and not data.get('anomalous')) else "❌"
            name = approach.replace('_', ' ').title()[:15]
            print(f"   {status} {name:15s} C:{data.get('compiled', False)} S:{data.get('simulated', False)} W:{data.get('has_waveform', False)} A:{data.get('anomalous', '-')}")
        
        return results
    
    def run_all_tests(self, circuits):
        """Run three-way tests on all circuits"""
        all_results = []
        start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"🚀 TESTING {len(circuits)} CIRCUITS with {self.model.upper()}")
        print(f"{'#'*70}")
        
        for i, circuit in enumerate(circuits, 1):
            result = self.test_three_ways(circuit)
            all_results.append(result)
            print(f"⏱️  Progress: {i}/{len(circuits)} ({i/len(circuits)*100:.0f}%)")
        
        total_time = time.time() - start_time
        print(f"\n✅ Complete in {total_time:.1f}s ({total_time/60:.1f}m)")
        
        return all_results


def calculate_statistics(results):
    """Calculate success rates"""
    stats = {
        'total_circuits': len(results),
        'prompt_only': {'correct': 0, 'total': 0},
        'mermaid_only': {'correct': 0, 'total': 0},
        'combined': {'correct': 0, 'total': 0}
    }
    
    for result in results:
        for approach in ['prompt_only', 'mermaid_only', 'combined']:
            data = result[approach]
            stats[approach]['total'] += 1
            
            is_correct = (
                data.get('compiled', False) and 
                data.get('simulated', False) and
                data.get('has_waveform', False) and
                not data.get('anomalous', False)
            )
            
            if is_correct:
                stats[approach]['correct'] += 1
    
    for approach in ['prompt_only', 'mermaid_only', 'combined']:
        total = stats[approach]['total']
        correct = stats[approach]['correct']
        stats[approach]['percentage'] = (correct / total * 100) if total > 0 else 0
    
    return stats


def save_results(results, stats, model, output_file=None):
    """Save results to JSON"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"multimodal_{model}_results_{timestamp}.json"
    
    output_data = {
        'metadata': {
            'test_date': datetime.now().isoformat(),
            'total_circuits': len(results),
            'model': model,
            'test_type': 'three_way_multimodal'
        },
        'statistics': stats,
        'results': results
    }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved: {output_path.name}")


def print_summary(stats, model):
    """Print summary"""
    print(f"\n{'='*70}")
    print(f"📊 {model.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Approach':<20} {'Correct':<10} {'Total':<10} {'Success %'}")
    print(f"{'-'*60}")
    
    for approach in ['prompt_only', 'mermaid_only', 'combined']:
        name = approach.replace('_', ' ').title()
        correct = stats[approach]['correct']
        total = stats[approach]['total']
        percentage = stats[approach]['percentage']
        print(f"{name:<20} {correct:<10} {total:<10} {percentage:>6.1f}%")
    
    winner = max(['prompt_only', 'mermaid_only', 'combined'], key=lambda x: stats[x]['percentage'])
    print(f"\n🏆 Winner: {winner.replace('_', ' ').title()} ({stats[winner]['percentage']:.1f}%)")


def main():
    """Main execution"""
    print("="*70)
    print("🚀 MULTIMODAL MERMAID TESTING")
    print("="*70)
    
    # Configuration
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
    MODELS = os.getenv("TEST_MODELS", "claude,gpt-4o").split(",")  # Test multiple models
    
    print(f"Backend: {BACKEND_URL}")
    print(f"Models: {', '.join(MODELS)}")
    
    # Load circuits once
    loader = MultimodalCircuitLoader()
    circuits = loader.get_all_circuits()
    print(f"✅ {len(circuits)} circuits loaded\n")
    
    all_model_results = {}
    
    # Test each model
    for model in MODELS:
        model = model.strip()
        print(f"\n{'='*70}")
        print(f"🔬 TESTING WITH: {model.upper()}")
        print(f"{'='*70}")
        
        tester = ThreeWayTester(backend_url=BACKEND_URL, model=model)
        results = tester.run_all_tests(circuits)
        stats = calculate_statistics(results)
        
        all_model_results[model] = {
            'results': results,
            'stats': stats
        }
        
        print_summary(stats, model)
        save_results(results, stats, model)
    
    # Compare models if multiple
    if len(MODELS) > 1:
        print(f"\n{'='*70}")
        print(f"🆚 MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"\n{'Model':<15} {'Prompt':<10} {'Mermaid':<10} {'Combined':<10} {'Average'}")
        print(f"{'-'*60}")
        
        for model in MODELS:
            model = model.strip()
            stats = all_model_results[model]['stats']
            prompt_pct = stats['prompt_only']['percentage']
            mermaid_pct = stats['mermaid_only']['percentage']
            combined_pct = stats['combined']['percentage']
            avg_pct = (prompt_pct + mermaid_pct + combined_pct) / 3
            
            print(f"{model:<15} {prompt_pct:>6.1f}%   {mermaid_pct:>6.1f}%   {combined_pct:>6.1f}%   {avg_pct:>6.1f}%")
        
        # Overall winner
        best_model = max(MODELS, key=lambda m: (
            all_model_results[m.strip()]['stats']['prompt_only']['percentage'] +
            all_model_results[m.strip()]['stats']['mermaid_only']['percentage'] +
            all_model_results[m.strip()]['stats']['combined']['percentage']
        ) / 3)
        
        print(f"\n🏆 Best Overall Model: {best_model.strip().upper()}")
    
    print("\n✅ All testing complete!")


if __name__ == "__main__":
    main()