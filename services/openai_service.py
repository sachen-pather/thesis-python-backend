# services/openai_service.py - FINAL FIXED VERSION
import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI Service with GPT-4o Support (2025 API - uses max_completion_tokens)"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
            raise RuntimeError("OPENAI_API_KEY not set")
        
        if not self.api_key.startswith("sk-"):
            logger.error(f"Invalid OpenAI API key format: {self.api_key[:10]}...")
            raise RuntimeError("Invalid OPENAI_API_KEY format (should start with 'sk-')")
        
        self.chat_url = "https://api.openai.com/v1/chat/completions"
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"OpenAI service initialized with key: {self.api_key[:7]}...")
    
    def invoke(self, prompt: str, model="gpt-4o", temperature=0.7, max_tokens=4000):
        """
        Call OpenAI Chat Completions API
        
        IMPORTANT: New API uses 'max_completion_tokens' instead of 'max_tokens'
        """
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        # Use max_completion_tokens for new API (2024+)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens  # Changed from max_tokens
        }
        
        try:
            logger.info(f"Calling OpenAI API with model: {model}")
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            logger.info(f"OpenAI API response status: {response.status_code}")
            
            # Check for errors
            if response.status_code == 401:
                raise RuntimeError("Invalid OpenAI API key (401 Unauthorized)")
            elif response.status_code == 429:
                raise RuntimeError("OpenAI rate limit exceeded (429)")
            elif response.status_code == 404:
                raise RuntimeError(f"Model '{model}' not found (404)")
            elif response.status_code >= 400:
                error_detail = response.json().get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"OpenAI API error ({response.status_code}): {error_detail}")
            
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.info(f"✅ OpenAI response received ({len(content)} chars)")
            return content
            
        except requests.exceptions.Timeout:
            raise RuntimeError("OpenAI API request timed out after 120s")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Failed to connect to OpenAI API")
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def test_connection(self):
        """Quick test to verify OpenAI API is working"""
        try:
            response = self.invoke("Say 'OpenAI is working!'", model="gpt-4o", max_tokens=50)
            return True, response
        except Exception as e:
            return False, str(e)


# ============================================================
# STANDALONE TEST SCRIPT
# ============================================================

if __name__ == "__main__":
    """
    Standalone test for OpenAI service
    Run: python services/openai_service.py
    """
    
    print("="*70)
    print("  OpenAI GPT-4o Service Test (2025 API)")
    print("="*70 + "\n")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    print(f"1. API Key Check:")
    print(f"   Set: {bool(api_key)}")
    print(f"   Length: {len(api_key)}")
    print(f"   Prefix: {api_key[:7]}..." if api_key else "   NOT SET")
    print(f"   Valid format: {api_key.startswith('sk-')}" if api_key else "   N/A")
    
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set!")
        print("   Add to .env file: OPENAI_API_KEY=sk-...")
        exit(1)
    
    # Test initialization
    print(f"\n2. Service Initialization:")
    try:
        service = OpenAIService()
        print("   ✅ Service initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        exit(1)
    
    # Test GPT-4o
    print(f"\n3. GPT-4o Connection Test:")
    try:
        response = service.invoke("Say 'GPT-4o is working!'", model="gpt-4o", max_tokens=50)
        print("   ✅ GPT-4o is working!")
        print(f"   Response: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ GPT-4o test failed: {e}")
        exit(1)
    
    # Test GPT-4o-mini
    print(f"\n4. GPT-4o-mini Test:")
    try:
        response = service.invoke("Say 'GPT-4o-mini works!'", model="gpt-4o-mini", max_tokens=50)
        print("   ✅ GPT-4o-mini is working!")
        print(f"   Response: {response[:100]}...")
    except Exception as e:
        print(f"   ⚠️  GPT-4o-mini test failed: {e}")
    
    # Test with Verilog generation
    print(f"\n5. Verilog Generation Test:")
    try:
        verilog_prompt = "Generate a simple 2-input AND gate in Verilog with testbench. Keep it under 200 characters."
        response = service.invoke(verilog_prompt, model="gpt-4o", max_tokens=500)
        print("   ✅ GPT-4o generated Verilog!")
        print(f"   Preview: {response[:150]}...")
    except Exception as e:
        print(f"   ⚠️  Verilog test failed: {e}")
    
    print("\n" + "="*70)
    print("  All Tests Complete! ✅")
    print("="*70)