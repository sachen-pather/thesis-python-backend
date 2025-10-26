# test_api_integrations.py
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def test_groq():
    """Test Groq API"""
    print("\n=== Testing Groq ===")
    try:
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not set")
            return False
        
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        response = llm.invoke("Say 'Groq works' in exactly 2 words")
        print(f"✅ Groq Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Groq Error: {str(e)}")
        return False

def test_claude():
    """Test Claude API"""
    print("\n=== Testing Claude ===")
    try:
        from services.claude_service import ClaudeService
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
        
        service = ClaudeService()
        response = service.invoke("Say 'Claude works' in exactly 2 words")
        print(f"✅ Claude Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Claude Error: {str(e)}")
        return False

def test_gemini():
    """Test Gemini API"""
    print("\n=== Testing Gemini ===")
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not set")
            return False
        
        genai.configure(api_key=api_key)
        # Use correct model name
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Say 'Gemini works' in exactly 2 words")
        print(f"✅ Gemini Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Gemini Error: {str(e)}")
        return False

def test_openai():
    """Test OpenAI API"""
    print("\n=== Testing OpenAI ===")
    try:
        from services.openai_service import OpenAIService
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            return False
        
        service = OpenAIService()
        response = service.invoke("Say 'OpenAI works' in exactly 2 words", model="gpt-4o-mini")
        print(f"✅ OpenAI Response: {response}")
        return True
    except Exception as e:
        print(f"❌ OpenAI Error: {str(e)}")
        return False

def test_unified_service():
    """Test Unified LLM Service"""
    print("\n=== Testing Unified Service ===")
    try:
        from services.unified_llm_service import UnifiedLLMService
        service = UnifiedLLMService()
        
        # Get available models
        models = service.get_available_models()
        print(f"Available models: {list(models.keys())}")
        
        # Test each available model
        results = {}
        for model_id in models.keys():
            try:
                response = service.invoke("Respond with 'OK'", model=model_id)
                results[model_id] = "✅ Working"
                print(f"  {model_id}: ✅ Working - {response[:50]}")
            except Exception as e:
                results[model_id] = f"❌ {str(e)[:50]}"
                print(f"  {model_id}: ❌ {str(e)[:50]}")
        
        return results
    except Exception as e:
        print(f"❌ Unified Service Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    print("🧪 Testing API Integrations\n")
    print("=" * 50)
    
    results = {
        "groq": test_groq(),
        "claude": test_claude(),
        "gemini": test_gemini(),
        "openai": test_openai()
    }
    
    print("\n" + "=" * 50)
    unified_results = test_unified_service()
    
    print("\n" + "=" * 50)
    print("\n📊 Summary:")
    print("-" * 50)
    
    for service, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {service.capitalize()}: {'Working' if status else 'Failed'}")
    
    if unified_results:
        print("\n🔧 Unified Service:")
        for model, status in unified_results.items():
            print(f"  {status} {model}")
    
    total_working = sum(1 for v in results.values() if v)
    print(f"\n📈 Working: {total_working}/{len(results)} services")