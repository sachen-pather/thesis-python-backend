# services/claude_service.py - Simple MVP
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class ClaudeService:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def invoke(self, prompt: str, model="claude-3-5-sonnet-20241022", max_tokens=4000):
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]