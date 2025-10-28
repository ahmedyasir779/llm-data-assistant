# Create model_checker.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Get available models
models = client.models.list()

print("🤖 Available Groq Models (Latest First):")
print("=" * 60)

for model in models.data:
    print(f"• {model.id}")
    if hasattr(model, 'created'):
        print(f"  Created: {model.created}")
    print()