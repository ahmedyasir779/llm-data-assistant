import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment
load_dotenv()


class SimpleLLM:
    def __init__(self):
        """Initialize Groq client"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError(" GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=api_key)
        self.model = os.getenv('MODEL_NAME', 'llama-3.3-70b-versatile')
        print(f"✓ Connected to Groq with {self.model}")
    
    def ask(self, question: str, system: str = None) -> str:
        """
        Ask a question, get an answer
        
        Args:
            question: Your question
            system: Optional instructions for the AI
            
        Returns:
            AI response as string
        """
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Add user question
        messages.append({"role": "user", "content": question})
        
        # Get response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def chat(self, conversation: List[Dict[str, str]]) -> str:
        """
        Multi-turn conversation
        
        Args:
            conversation: List of {"role": "user/assistant", "content": "..."}
            
        Returns:
            AI response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    

# Quick test
if __name__ == "__main__":
    print("🧪 Testing Groq connection...\n")
    
    llm = SimpleLLM()
    
    # Test 1: Simple question
    question = "What is machine learning in one sentence?"
    print(f"Q: {question}")
    
    answer = llm.ask(question)
    print(f"A: {answer}\n")
    
    # Test 2: With system message
    system = "You are a helpful data analyst. Keep answers concise."
    question = "What's the best way to clean messy data?"
    print(f"System: {system}")
    print(f"Q: {question}")
    
    answer = llm.ask(question, system=system)
    print(f"A: {answer}\n")
    
    print(" Groq is working perfectly!")