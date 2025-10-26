# services/api_key_manager.py
import os
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from langchain_groq import ChatGroq
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class APIKeyStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID = "invalid"
    NETWORK_ERROR = "network_error"

@dataclass
class APIKeyConfig:
    key: str
    name: str
    is_free: bool = True
    max_requests_per_minute: int = 30
    max_requests_per_day: int = 14400
    priority: int = 1  # Lower number = higher priority
    cooldown_minutes: int = 60
    last_used: float = 0
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    daily_usage: int = 0
    last_reset: float = 0
    is_gemini: bool = False

class APIKeyManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_keys: List[APIKeyConfig] = []
        self.current_key_index = 0
        self.load_api_keys()
        
    def load_api_keys(self):
        """Load API keys from environment variables"""
        # Primary API key (paid/premium)
        primary_key = os.getenv("GROQ_API_KEY")
        if primary_key:
            self.api_keys.append(APIKeyConfig(
                key=primary_key,
                name="Primary",
                is_free=False,
                max_requests_per_minute=100,
                max_requests_per_day=100000,
                priority=0
            ))
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.api_keys.append(APIKeyConfig(
                key=gemini_key,
                name="Gemini-2.0-Flash",
                is_free=True,
                is_gemini=True,  # Mark as Gemini key
                max_requests_per_minute=60,
                max_requests_per_day=100000,
                priority=0.5  # Higher priority than free keys
            ))      
        
        # Fallback API keys (free tiers)
        fallback_keys = [
            ("GROQ_API_KEY_FALLBACK_1", "Fallback-1"),
            ("GROQ_API_KEY_FALLBACK_2", "Fallback-2"),
            ("GROQ_API_KEY_FALLBACK_3", "Fallback-3"),
            ("GROQ_API_KEY_FREE", "Free-Tier"),
        ]
        
        for env_var, name in fallback_keys:
            key = os.getenv(env_var)
            if key:
                self.api_keys.append(APIKeyConfig(
                    key=key,
                    name=name,
                    is_free=True,
                    max_requests_per_minute=30,
                    max_requests_per_day=14400,  # Typical free tier limit
                    priority=len(self.api_keys) + 1
                ))
        
        # Sort by priority
        self.api_keys.sort(key=lambda x: x.priority)
        
        if not self.api_keys:
            raise ValueError("No API keys found! Please set GROQ_API_KEY or fallback keys.")
        
        self.logger.info(f"Loaded {len(self.api_keys)} API keys")
        for key in self.api_keys:
            self.logger.info(f"  - {key.name}: {'Free' if key.is_free else 'Paid'} tier")

    def get_active_key(self) -> Optional[APIKeyConfig]:
        """Get the currently active API key"""
        current_time = time.time()
        
        # Reset daily usage if needed
        for key in self.api_keys:
            if current_time - key.last_reset > 86400:  # 24 hours
                key.daily_usage = 0
                key.last_reset = current_time
                if key.status == APIKeyStatus.QUOTA_EXCEEDED:
                    key.status = APIKeyStatus.ACTIVE
        
        # Find the best available key
        for key in self.api_keys:
            if self._is_key_available(key, current_time):
                return key
        
        # If no keys available, try to reset rate-limited keys
        self._reset_rate_limited_keys(current_time)
        
        # Try again after reset
        for key in self.api_keys:
            if self._is_key_available(key, current_time):
                return key
        
        return None

    def _is_key_available(self, key: APIKeyConfig, current_time: float) -> bool:
        """Check if an API key is available for use"""
        if key.status in [APIKeyStatus.INVALID]:
            return False
        
        # Check cooldown period
        if key.status == APIKeyStatus.RATE_LIMITED:
            if current_time - key.last_used < (key.cooldown_minutes * 60):
                return False
            else:
                key.status = APIKeyStatus.ACTIVE
        
        # Check daily quota
        if key.daily_usage >= key.max_requests_per_day:
            key.status = APIKeyStatus.QUOTA_EXCEEDED
            return False
        
        return True

    def _reset_rate_limited_keys(self, current_time: float):
        """Reset rate-limited keys that have passed cooldown"""
        for key in self.api_keys:
            if (key.status == APIKeyStatus.RATE_LIMITED and 
                current_time - key.last_used >= (key.cooldown_minutes * 60)):
                key.status = APIKeyStatus.ACTIVE
                self.logger.info(f"Reset rate limit for key: {key.name}")

    def mark_key_used(self, key: APIKeyConfig):
        """Mark a key as used and update usage statistics"""
        current_time = time.time()
        key.last_used = current_time
        key.daily_usage += 1
        
        # Check if approaching limits
        if key.daily_usage >= key.max_requests_per_day * 0.9:
            self.logger.warning(f"Key {key.name} approaching daily limit: {key.daily_usage}/{key.max_requests_per_day}")

    def mark_key_error(self, key: APIKeyConfig, error: Exception):
        """Mark a key as having an error and determine appropriate status"""
        error_msg = str(error).lower()
        
        if "rate limit" in error_msg or "too many requests" in error_msg:
            key.status = APIKeyStatus.RATE_LIMITED
            self.logger.warning(f"Rate limit hit for key {key.name}, cooling down for {key.cooldown_minutes} minutes")
        elif "quota" in error_msg or "exceeded" in error_msg:
            key.status = APIKeyStatus.QUOTA_EXCEEDED
            self.logger.error(f"Quota exceeded for key {key.name}")
        elif "invalid" in error_msg or "unauthorized" in error_msg:
            key.status = APIKeyStatus.INVALID
            self.logger.error(f"Invalid API key: {key.name}")
        else:
            key.status = APIKeyStatus.NETWORK_ERROR
            self.logger.error(f"Network error for key {key.name}: {error}")

    def get_status_report(self) -> Dict[str, Any]:
        """Get status report of all API keys"""
        current_time = time.time()
        report = {
            "total_keys": len(self.api_keys),
            "active_keys": 0,
            "keys": []
        }
        
        for key in self.api_keys:
            key_info = {
                "name": key.name,
                "status": key.status.value,
                "is_free": key.is_free,
                "daily_usage": key.daily_usage,
                "daily_limit": key.max_requests_per_day,
                "usage_percentage": (key.daily_usage / key.max_requests_per_day) * 100,
                "available": self._is_key_available(key, current_time)
            }
            
            if key.status == APIKeyStatus.RATE_LIMITED:
                cooldown_remaining = (key.cooldown_minutes * 60) - (current_time - key.last_used)
                key_info["cooldown_remaining_minutes"] = max(0, cooldown_remaining / 60)
            
            report["keys"].append(key_info)
            
            if key_info["available"]:
                report["active_keys"] += 1
        
        return report

# Enhanced services with fallback support
class EnhancedVerilogService:
    def __init__(self):
        self.api_manager = APIKeyManager()
        self.logger = logging.getLogger(__name__)
        self.compilation_log = ""
        self._current_llm = None
        
    def _get_llm(self) -> Union[ChatGroq, Any]:
        """Get LLM instance with current active API key"""
        active_key = self.api_manager.get_active_key()
        
        if not active_key:
            raise Exception("No API keys available!")
        
        # Handle Gemini keys
        if active_key.is_gemini and GEMINI_AVAILABLE:
            genai.configure(api_key=active_key.key)
            return genai.GenerativeModel('gemini-2.0-flash')
        
        # Handle Groq keys (existing code)
        if not self._current_llm or self._current_llm.api_key != active_key.key:
            self._current_llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                api_key=active_key.key,
                temperature=0.1,
                max_retries=1
            )
        
        return self._current_llm

    def generate_verilog_with_fallback(self, mermaid_code: str, process_description: str = "") -> str:
        """Generate Verilog with automatic fallback between API keys"""
        max_attempts = len(self.api_manager.api_keys)
        
        for attempt in range(max_attempts):
            try:
                active_key = self.api_manager.get_active_key()
                if not active_key:
                    raise Exception("No API keys available")
                
                llm = self._get_llm()
                
                # Gemini-specific invocation
                if active_key.is_gemini and GEMINI_AVAILABLE:
                    response = llm.generate_content(self._build_verilog_prompt(mermaid_code, process_description))
                    verilog_code = response.text.strip()
                # Groq invocation
                else:
                    response = llm.invoke(self._build_verilog_prompt(mermaid_code, process_description))
                    verilog_code = response.content.strip()
                
                # Mark successful usage
                self.api_manager.mark_key_used(active_key)
                
                # Process and return the code
                return self._process_verilog_code(verilog_code)
                
            except Exception as e:
                current_key = self.api_manager.get_active_key()
                if current_key:
                    self.api_manager.mark_key_error(current_key, e)
                
                self.log_message(f"Attempt {attempt + 1} failed: {str(e)}", "error")
                
                # If this was the last attempt, re-raise
                if attempt == max_attempts - 1:
                    raise Exception(f"All API keys failed. Last error: {str(e)}")
                
                # Wait briefly before next attempt
                time.sleep(1)
        
        raise Exception("Fallback generation failed - no working API keys")

    def _build_verilog_prompt(self, mermaid_code: str, process_description: str) -> str:
        """Build the Verilog generation prompt"""
        return f"""
        Convert this Mermaid diagram to valid Verilog-2001 code that compiles with iverilog:
        {mermaid_code}
        
        Process Description: {process_description if process_description else "Simple pass-through with 1-clock delay"}
        
        CRITICAL SYNTAX REQUIREMENTS:
        1. Use ONLY standard Verilog-2001 keywords and syntax
        2. Use 'reg' for registers, 'wire' for nets
        3. All module ports must be properly declared with directions
        4. NO SystemVerilog constructs (logic, always_ff, etc.)
        5. Use proper always block sensitivity lists
        6. Avoid reserved keywords as signal names (like 'output')
        7. Use proper blocking/non-blocking assignments
        
        OUTPUT: Return ONLY the complete, valid Verilog code. No explanations, no markdown blocks.
        """

    def _process_verilog_code(self, verilog_code: str) -> str:
        """Process and clean the generated Verilog code"""
        # Clean up markdown artifacts
        if "```" in verilog_code:
            lines = verilog_code.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not any(line.strip().startswith(x) for x in ['```', '#', '*', '-'])):
                    code_lines.append(line)
            
            verilog_code = '\n'.join(code_lines).strip()
        
        # Add timescale if not present
        if "`timescale" not in verilog_code and "timescale" not in verilog_code.lower():
            verilog_code = "`timescale 1ns/1ps\n\n" + verilog_code
        
        return verilog_code

    def log_message(self, message: str, level: str = "info"):
        """Add timestamped log message"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.compilation_log += f"[{timestamp}] {level.upper()}: {message}\n"

    def get_api_status(self) -> Dict[str, Any]:
        """Get API key status for monitoring"""
        return self.api_manager.get_status_report()

class EnhancedMermaidService:
    def __init__(self):
        self.api_manager = APIKeyManager()
        self.logger = logging.getLogger(__name__)
        self._current_llm = None

    def _get_llm(self) -> ChatGroq:
        """Get LLM instance with current active API key"""
        active_key = self.api_manager.get_active_key()
        
        if not active_key:
            raise Exception("No API keys available!")
        
        if not self._current_llm or self._current_llm.api_key != active_key.key:
            self._current_llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                api_key=active_key.key,
                temperature=0.1,
                max_retries=1
            )
            self.logger.info(f"Mermaid service using key: {active_key.name}")
        
        return self._current_llm

    def generate_mermaid_with_fallback(self, user_prompt: str) -> str:
        """Generate Mermaid diagram with API key fallback"""
        max_attempts = len(self.api_manager.api_keys)
        
        for attempt in range(max_attempts):
            try:
                active_key = self.api_manager.get_active_key()
                if not active_key:
                    raise Exception("No API keys available")
                
                llm = self._get_llm()
                
                mermaid_prompt = f"""
                Convert this natural language description into a Mermaid diagram suitable for digital hardware design:
                
                USER REQUEST: "{user_prompt}"
                
                **GUIDELINES:**
                1. Use 'graph TD' (top-down) or 'graph LR' (left-right) syntax
                2. Create nodes representing hardware components/blocks
                3. Use arrows to show data flow or signal paths
                4. Include meaningful labels for inputs, outputs, and processing blocks
                5. Keep it simple but accurate for hardware implementation
                
                **OUTPUT:** Return ONLY the Mermaid diagram code, no explanations or markdown blocks.
                
                Example format:
                graph TD
                    CLK[Clock] --> PROC[Processing Unit]
                    RST[Reset] --> PROC
                    A[Input Data] --> PROC
                    PROC --> C[Output Register]
                    CLK --> C
                """
                
                response = llm.invoke(mermaid_prompt)
                mermaid_code = response.content.strip()
                
                # Mark successful usage
                self.api_manager.mark_key_used(active_key)
                
                # Clean up any markdown artifacts
                if "```" in mermaid_code:
                    lines = mermaid_code.split('\n')
                    code_lines = []
                    in_code_block = False
                    
                    for line in lines:
                        if line.strip().startswith('```'):
                            in_code_block = not in_code_block
                            continue
                        if in_code_block or (not any(line.strip().startswith(x) for x in ['```', '#', '*', '-'])):
                            code_lines.append(line)
                    
                    mermaid_code = '\n'.join(code_lines).strip()
                
                return mermaid_code
                
            except Exception as e:
                current_key = self.api_manager.get_active_key()
                if current_key:
                    self.api_manager.mark_key_error(current_key, e)
                
                if attempt == max_attempts - 1:
                    raise Exception(f"All API keys failed for Mermaid generation. Last error: {str(e)}")
                
                time.sleep(1)
        
        raise Exception("Mermaid generation failed - no working API keys")

# Enhanced Analysis Service with fallback
class EnhancedAnalysisService:
    def __init__(self):
        self.api_manager = APIKeyManager()
        self.logger = logging.getLogger(__name__)
        self._current_llm = None

    def _get_llm(self) -> ChatGroq:
        """Get LLM instance with current active API key"""
        active_key = self.api_manager.get_active_key()
        
        if not active_key:
            raise Exception("No API keys available!")
        
        if not self._current_llm or self._current_llm.api_key != active_key.key:
            self._current_llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                api_key=active_key.key,
                temperature=0.1,
                max_retries=1
            )
        
        return self._current_llm

    def analyze_waveform_with_fallback(self, csv_data: str, verilog_code: str = "") -> str:
        """Analyze waveform with API key fallback"""
        max_attempts = len(self.api_manager.api_keys)
        
        for attempt in range(max_attempts):
            try:
                active_key = self.api_manager.get_active_key()
                if not active_key:
                    raise Exception("No API keys available")
                
                llm = self._get_llm()
                
                # Build analysis prompt
                lines = csv_data.split('\n')
                sample_data = '\n'.join(lines[:100])
                
                prompt = f"""
                Analyze this Verilog simulation waveform data for potential issues:
                
                **WAVEFORM DATA:**
                ```
                {sample_data}
                ```
                
                **ANALYSIS REQUIREMENTS:**
                1. Check for signal integrity issues
                2. Analyze timing behavior
                3. Verify functional correctness
                4. Identify any critical issues
                5. Provide actionable recommendations
                
                **OUTPUT FORMAT:** Use markdown with clear sections and provide specific findings.
                """
                
                response = llm.invoke(prompt)
                
                # Mark successful usage
                self.api_manager.mark_key_used(active_key)
                
                return response.content
                
            except Exception as e:
                current_key = self.api_manager.get_active_key()
                if current_key:
                    self.api_manager.mark_key_error(current_key, e)
                
                if attempt == max_attempts - 1:
                    raise Exception(f"All API keys failed for analysis. Last error: {str(e)}")
                
                time.sleep(1)
        
        raise Exception("Waveform analysis failed - no working API keys")