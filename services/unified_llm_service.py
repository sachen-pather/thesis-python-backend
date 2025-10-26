# services/unified_llm_service.py - UPDATED FOR GPT-5
from typing import Optional, Dict, Any
from langchain_groq import ChatGroq
import os
import logging

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from .claude_service import ClaudeService
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    from .openai_service import OpenAIService
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnifiedLLMService:
    """Unified service for Groq, Claude, Gemini, and OpenAI GPT-5"""
    
    def __init__(self):
        # Initialize Groq
        self.groq_service = None
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.groq_service = ChatGroq(
                    model="llama-3.3-70b-versatile", 
                    api_key=groq_api_key
                )
                logger.info("✅ Groq service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {str(e)}")
        
        # Initialize Claude
        self.claude_service = None
        if CLAUDE_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.claude_service = ClaudeService()
                logger.info("✅ Claude service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {str(e)}")
        
        # Initialize Gemini
        self.gemini_service = None
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_service = genai
                logger.info("✅ Gemini service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {str(e)}")
        
        # Initialize OpenAI (GPT-5 support)
        self.openai_service = None
        self.openai_init_error = None
        
        if OPENAI_AVAILABLE:
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            
            if not openai_key:
                self.openai_init_error = "OPENAI_API_KEY not set in environment"
                logger.warning(self.openai_init_error)
            elif not openai_key.startswith("sk-"):
                self.openai_init_error = f"Invalid API key format"
                logger.error(self.openai_init_error)
            else:
                try:
                    self.openai_service = OpenAIService()
                    logger.info("✅ OpenAI service initialized (GPT-5 support)")
                except Exception as e:
                    self.openai_init_error = str(e)
                    logger.error(f"❌ Failed to initialize OpenAI: {e}")
        else:
            self.openai_init_error = "OpenAI service not available"
            logger.warning(self.openai_init_error)
        
        # Model categories (updated with GPT-5)
        self.model_categories = {
            "verilog_generation": ["gpt-5", "claude", "gpt-4o", "groq", "gemini/gemini-2.0-flash-exp"],
            "analysis": ["gpt-5", "claude", "gpt-4o", "groq", "gemini/gemini-2.0-flash-exp"],
            "mermaid_generation": ["gpt-5", "claude", "groq", "gemini/gemini-2.0-flash-exp"],
            "waveform_analysis": ["gpt-5", "claude", "gpt-4o", "groq"],
            "complex_reasoning": ["gpt-5", "claude", "gpt-4o"],
            "general": ["gpt-5", "claude", "gpt-4o", "groq", "gemini/gemini-2.0-flash-exp"]
        }
    
    def invoke(self, prompt: str, model: str = "groq", **kwargs) -> str:
        """Invoke any available model"""
        
        if model == "groq":
            if not self.groq_service:
                raise RuntimeError("Groq service not available")
            return self.groq_service.invoke(prompt).content
        
        elif model == "claude":
            if not self.claude_service:
                raise RuntimeError("Claude service not available")
            return self.claude_service.invoke(prompt, **kwargs)
        
        elif model.startswith("gemini/"):
            if not self.gemini_service:
                raise RuntimeError("Gemini service not available")
            try:
                model_name = model.split('/')[1]
                gemini_model = self.gemini_service.GenerativeModel(model_name)
                response = gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                raise RuntimeError(f"Gemini error: {str(e)}")
        
        elif model in ["gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4"]:
            if not self.openai_service:
                raise RuntimeError(f"OpenAI service not available: {self.openai_init_error}")
            return self.openai_service.invoke(prompt, model=model, **kwargs)
        
        else:
            raise RuntimeError(f"Unknown model: {model}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get all available models including GPT-5"""
        models = {}
        
        if self.groq_service:
            models["groq"] = {
                "id": "groq",
                "name": "Llama 3.3 70B Versatile",
                "description": "Fast and reliable model via Groq",
                "provider": "Groq",
                "supports_vision": False,
                "context_length": 32768,
                "best_for": ["general", "fast_response", "verilog"]
            }
        
        if self.claude_service:
            models["claude"] = {
                "id": "claude",
                "name": "Claude 3.5 Sonnet",
                "description": "Anthropic's Claude for complex reasoning",
                "provider": "Anthropic",
                "supports_vision": False,
                "context_length": 200000,
                "best_for": ["complex_reasoning", "analysis", "verilog"]
            }
        
        if self.gemini_service:
            models["gemini/gemini-2.0-flash-exp"] = {
                "id": "gemini/gemini-2.0-flash-exp",
                "name": "Gemini 2.0 Flash",
                "provider": "Google",
                "description": "Fast and efficient model",
                "supports_vision": True,
                "context_length": 1000000,
                "best_for": ["general", "fast_response", "vision"]
            }
        
        # Add OpenAI models (GPT-5, GPT-4o, etc.) if service is available
        if self.openai_service:
            models.update({
                "gpt-5": {
                    "id": "gpt-5",
                    "name": "GPT-5",
                    "provider": "OpenAI",
                    "description": "OpenAI's most advanced model (2025)",
                    "supports_vision": True,
                    "context_length": 200000,
                    "best_for": ["complex_reasoning", "analysis", "verilog", "everything"],
                    "is_latest": True
                },
                "gpt-4o": {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "provider": "OpenAI",
                    "description": "OpenAI's GPT-4 optimized model",
                    "supports_vision": True,
                    "context_length": 128000,
                    "best_for": ["complex_tasks", "analysis", "vision"]
                },
                "gpt-4o-mini": {
                    "id": "gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "provider": "OpenAI",
                    "description": "Fast and cost-effective GPT-4 model",
                    "supports_vision": False,
                    "context_length": 128000,
                    "best_for": ["general", "fast_response", "verilog"]
                }
            })
        
        return models
    
    def get_recommended_models(self, task_type: str) -> list:
        """Get recommended models for a task (GPT-5 first for most tasks)"""
        return self.model_categories.get(task_type, self.model_categories["general"])
    
    def test_model(self, model: str) -> Dict[str, Any]:
        """Test if a model is working"""
        
        # Check if OpenAI models are available
        if model in ["gpt-5", "gpt-4o", "gpt-4o-mini"]:
            if not self.openai_service:
                return {
                    "success": False,
                    "model": model,
                    "error": f"OpenAI service not available: {self.openai_init_error}",
                    "suggestion": "Check if OPENAI_API_KEY is set in .env file"
                }
        
        try:
            response = self.invoke("Say 'test successful'", model=model)
            return {
                "success": True,
                "model": model,
                "response": response[:100] + "..." if len(response) > 100 else response
            }
        except Exception as e:
            return {
                "success": False,
                "model": model,
                "error": str(e),
                "suggestion": self._get_error_suggestion(model, str(e))
            }
    
    def _get_error_suggestion(self, model: str, error: str) -> str:
        """Provide helpful suggestions based on error"""
        if "401" in error or "Unauthorized" in error:
            return "Check if your API key is valid and correctly set in .env"
        elif "404" in error:
            return f"Model '{model}' may not be available with your API key tier"
        elif "429" in error:
            return "Rate limit exceeded - wait a few seconds and try again"
        elif "timeout" in error.lower():
            return "API request timed out - check your internet connection"
        elif "not set" in error.lower() or "not available" in error.lower():
            return f"Add the required API key to your .env file"
        else:
            return "Check API key and network connection"