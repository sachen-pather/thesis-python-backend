import requests
import json
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables from .env in project root
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    logger.warning(".env file not found in project root")

class OpenRouterService:
    """Service for interacting with OpenRouter API models"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.site_url = os.getenv("SITE_URL", "http://localhost:3000")
        self.site_name = os.getenv("SITE_NAME", "Verilog Generation System")
        
        # Available free models
        self.available_models = {
            "qwen-2.5-coder-32b": {
                "id": "qwen/qwen-2.5-coder-32b-instruct:free",
                "name": "Qwen2.5 Coder 32B Instruct",
                "description": "Specialized coding model, excellent for Verilog generation",
                "provider": "Qwen",
                "supports_vision": False,
                "context_length": 32768,
                "best_for": ["code_generation", "verilog", "hardware_design"]
            },
            "kimi-dev-72b": {
                "id": "moonshotai/kimi-dev-72b:free",
                "name": "Kimi Dev 72B",
                "description": "Large development-focused model",
                "provider": "MoonshotAI",
                "supports_vision": False,
                "context_length": 200000,
                "best_for": ["complex_reasoning", "analysis", "documentation"]
            },
            "devstral-small-2505": {
                "id": "mistralai/devstral-small-2505:free",
                "name": "Devstral Small 2505",
                "description": "Mistral's development-focused model",
                "provider": "Mistral",
                "supports_vision": False,
                "context_length": 32768,
                "best_for": ["code_generation", "debugging", "optimization"]
            },
            "deepseek-r1-0528": {
                "id": "deepseek/deepseek-r1-0528:free",
                "name": "DeepSeek R1 0528",
                "description": "Reasoning-focused model with chain-of-thought",
                "provider": "DeepSeek",
                "supports_vision": False,
                "context_length": 32768,
                "best_for": ["reasoning", "analysis", "problem_solving"]
            },
            "kimi-k2": {
                "id": "moonshotai/kimi-k2:free",
                "name": "Kimi K2",
                "description": "Advanced reasoning and analysis model",
                "provider": "MoonshotAI",
                "supports_vision": False,
                "context_length": 200000,
                "best_for": ["analysis", "verification", "optimization"]
            },
            "gemma-3-27b": {
                "id": "google/gemma-3-27b-it:free",
                "name": "Gemma 3 27B",
                "description": "Google's instruction-tuned model with vision support",
                "provider": "Google",
                "supports_vision": True,
                "context_length": 8192,
                "best_for": ["multimodal", "analysis", "general_tasks"]
            },
            "mistral-small-3.1-24b": {
                "id": "mistralai/mistral-small-3.1-24b-instruct:free",
                "name": "Mistral Small 3.1 24B",
                "description": "Balanced model with vision capabilities",
                "provider": "Mistral",
                "supports_vision": True,
                "context_length": 32768,
                "best_for": ["multimodal", "code_generation", "analysis"]
            }
        }
    
    
    def _validate_api_key(self):
        """Validate API key and raise error if missing"""
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key not configured. "
                "Please set OPENROUTER_API_KEY in your environment variables."
            )    
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        
    
    def invoke(self, prompt: str, model_key: str = "qwen-2.5-coder-32b", 
               temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Invoke OpenRouter model with a prompt
        
        Args:
            prompt: The input prompt
            model_key: Key for the model to use (from available_models)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response content
        """
        self._validate_api_key()
        
        if model_key not in self.available_models:
            raise ValueError(f"Model '{model_key}' not available. Available models: {list(self.available_models.keys())}")
        
        model_id = self.available_models[model_key]["id"]
        
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.get_headers(),
                json=payload,  # Use json parameter for automatic serialization
                timeout=60
            )
            
            # Handle 401 Unauthorized specifically
            if response.status_code == 401:
                raise PermissionError(
                    f"OpenRouter authentication failed (401): {response.json().get('error', {}).get('message', 'Invalid API key')}"
                )
            
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                raise ValueError("No response generated from OpenRouter")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.RequestException as e:
            # Log detailed error information
            logger.error(f"OpenRouter request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise RuntimeError(f"OpenRouter request failed: {str(e)}")
        except Exception as e:
            logger.error(f"OpenRouter invocation failed: {str(e)}")
            raise RuntimeError(f"OpenRouter invocation failed: {str(e)}")   
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models with metadata"""
        return self.available_models
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return self.available_models.get(model_key)
    
    def test_connection(self, model_key: str = "qwen-2.5-coder-32b") -> Dict[str, Any]:
        """Test connection to OpenRouter with a simple prompt"""
        try:
            # First validate API key
            self._validate_api_key()
            
            # Then test actual connection
            response = self.invoke("Hello, can you confirm you're working?", model_key)
            return {
                "success": True,
                "model": model_key,
                "response": response
            }
        except Exception as e:
            return {
                "success": False,
                "model": model_key,
                "error": str(e)
            }
