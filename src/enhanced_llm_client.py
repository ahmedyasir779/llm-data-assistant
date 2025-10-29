import os
import time
from typing import Optional, Dict, Any, List
from groq import Groq
import streamlit as st


class EnhancedLLMClient:
    """
    Enhanced LLM client with advanced error handling and retry logic
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """Initialize enhanced LLM client"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Send chat request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout
                )
                
                content = response.choices[0].message.content
                return self._clean_response(content)
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    st.warning(f"Retry {attempt + 1}/{self.max_retries}... waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    return self._handle_error(e)
        
        return "Unable to get response after multiple retries."
    
    def _clean_response(self, text: str) -> str:
        """Clean LLM response from common issues"""
        gibberish_patterns = ["�", "\\x", "<|", "|>", "[INST]", "[/INST]"]
        
        for pattern in gibberish_patterns:
            text = text.replace(pattern, "")
        
        text = " ".join(text.split())
        
        if text and not text[-1] in ".!?":
            last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
        
        return text.strip()
    
    def _handle_error(self, error: Exception) -> str:
        """Handle errors gracefully with helpful messages"""
        error_msg = str(error).lower()
        
        if "rate limit" in error_msg:
            return "⚠️ **Rate limit reached**\n\nPlease wait a moment before trying again.\n\n💡 Tip: Try asking simpler questions or wait 60 seconds."
        
        elif "timeout" in error_msg:
            return "⏱️ **Request timed out**\n\nThe model took too long to respond.\n\n💡 Try:\n- Asking a simpler question\n- Using a smaller dataset"
        
        elif "api key" in error_msg or "authentication" in error_msg:
            return "🔑 **API Key Issue**\n\nThere's a problem with your Groq API key.\n\n💡 Check:\n- Key is set in .env file\n- Key is valid at console.groq.com"
        
        else:
            return f"❌ **Unexpected Error**\n\nSomething went wrong: {error}\n\n💡 Try refreshing the page"
    
    def analyze_data(self, data_info: Dict[str, Any], question: str) -> str:
        """Analyze data and answer questions"""
        context = self._create_data_context(data_info)
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful data analysis assistant.

Guidelines:
- Give clear, concise answers
- Use bullet points for lists
- Mention specific numbers when relevant
- Suggest visualizations when appropriate
- Be conversational and friendly"""
            },
            {
                "role": "user",
                "content": f"""Dataset Information:
{context}

Question: {question}

Please provide a clear, helpful answer."""
            }
        ]
        
        return self.chat(messages)
    
    def _create_data_context(self, data_info: Dict[str, Any]) -> str:
        """Create formatted context from data information"""
        context_parts = []
        
        if "shape" in data_info:
            rows, cols = data_info["shape"]
            context_parts.append(f"- Dataset: {rows:,} rows × {cols} columns")
        
        if "columns" in data_info:
            context_parts.append(f"- Columns: {', '.join(data_info['columns'])}")
        
        if "sample" in data_info:
            context_parts.append(f"- Sample data:\n{data_info['sample']}")
        
        if "stats" in data_info:
            context_parts.append(f"- Statistics:\n{data_info['stats']}")
        
        return "\n".join(context_parts)