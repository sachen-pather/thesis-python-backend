"""
Test individual model APIs
Tests GPT-4o and Claude to verify they work before running full comparison

SAVE AS: tests/unit/test_individual_models.py
RUN: python tests/unit/test_individual_models.py
   OR: python -m tests.unit.test_individual_models
"""

import sys
import os

# Add project root to path so we can import services
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from services.unified_llm_service import UnifiedLLMService

def test_models():
    print("=" * 60)
    print("🧪 Testing Individual LLM Models")
    print("=" * 60)
    
    llm = UnifiedLLMService()
    
    results = {}
    
    # Test GPT-4o
    print("\n1️⃣  Testing GPT-4o...")
    print("   Sending: 'Say exactly: GPT-4o is working!'")
    try:
        response = llm.invoke(
            "Say exactly: 'GPT-4o is working!' and nothing else.", 
            model="gpt-4o"
        )
        print(f"   ✅ GPT-4o Response: {response[:150]}")
        results['gpt-4o'] = {'success': True, 'response': response[:100]}
    except Exception as e:
        print(f"   ❌ GPT-4o Failed: {e}")
        results['gpt-4o'] = {'success': False, 'error': str(e)}
    
    # Test Claude
    print("\n2️⃣  Testing Claude Sonnet 4.5...")
    print("   Sending: 'Say exactly: Claude is working!'")
    try:
        response = llm.invoke(
            "Say exactly: 'Claude is working!' and nothing else.", 
            model="claude"
        )
        print(f"   ✅ Claude Response: {response[:150]}")
        results['claude'] = {'success': True, 'response': response[:100]}
    except Exception as e:
        print(f"   ❌ Claude Failed: {e}")
        results['claude'] = {'success': False, 'error': str(e)}
    
    # Test with a Verilog analysis prompt (more realistic)
    print("\n3️⃣  Testing with Verilog analysis prompt...")
    test_analysis = """
Analyze this simple waveform data:
time,signal,value
0,clk,0
10,clk,1
20,clk,0
30,clk,1

Is this a valid clock signal? Answer in one sentence.
"""
    
    for model in ['gpt-4o', 'claude']:
        if results[model]['success']:
            print(f"\n   Testing {model} with analysis prompt...")
            try:
                response = llm.invoke(test_analysis, model=model)
                print(f"   ✅ {model} analysis: {response[:100]}...")
                results[model]['analysis_works'] = True
            except Exception as e:
                print(f"   ⚠️  {model} analysis failed: {e}")
                results[model]['analysis_works'] = False
    
    # Show available models
    print("\n" + "=" * 60)
    print("📋 Available Models in System:")
    print("=" * 60)
    models = llm.get_available_models()
    for model_id, info in models.items():
        status = "✅" if model_id in results and results[model_id]['success'] else "❓"
        print(f"   {status} {info['name']} ({model_id})")
        print(f"      Provider: {info['provider']}")
        if model_id in results and not results[model_id]['success']:
            print(f"      Error: {results[model_id].get('error', 'Unknown')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print("=" * 60)
    
    working_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"   Working models: {working_count}/{total_count}")
    
    if working_count == total_count:
        print("   🎉 All models working! Ready to run comparison.")
        print("\n   Next steps:")
        print("   1. Apply the main.py fix: python fix_model_routing.py")
        print("   2. Restart backend: uvicorn main:app --reload")
        print("   3. Run comparison: python tests/analysis/gpt_claude_only_comparison.py")
    elif working_count > 0:
        working = [k for k, v in results.items() if v['success']]
        print(f"   ⚠️  Only {', '.join(working)} working.")
        print("   Check your API keys in .env file:")
        if 'gpt-4o' in results and not results['gpt-4o']['success']:
            print("      - OPENAI_API_KEY=sk-...")
        if 'claude' in results and not results['claude']['success']:
            print("      - ANTHROPIC_API_KEY=sk-ant-...")
    else:
        print("   ❌ No models working! Check all API keys in .env")
        print("   Required in .env:")
        print("      OPENAI_API_KEY=sk-...")
        print("      ANTHROPIC_API_KEY=sk-ant-...")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    test_models()