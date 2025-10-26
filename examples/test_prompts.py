import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now imports work cleanly
from src import SimpleLLM, PromptTemplates


def test_temperature_effects():
    """Test how temperature affects outputs"""
    print("\n" + "=" * 60)
    print("TEST 1: Temperature Effects")
    print("=" * 60)
    
    llm = SimpleLLM()
    
    question = "List 3 creative names for a data analysis startup."
    
    # Test different temperatures
    from groq import Groq
    import os
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    for temp in [0.0, 0.7, 1.5]:
        print(f"\n Temperature: {temp}")
        print(f"Q: {question}")
        
        response = client.chat.completions.create(
            model=os.getenv('MODEL_NAME'),
            messages=[{"role": "user", "content": question}],
            temperature=temp,
            max_tokens=200
        )
        
        print(f"A: {response.choices[0].message.content}\n")


def test_system_prompts():
    """Test different system prompts"""
    print("\n" + "=" * 60)
    print("TEST 2: System Prompt Effectiveness")
    print("=" * 60)
    
    llm = SimpleLLM()
    templates = PromptTemplates()
    
    question = "What should I do with missing data in my dataset?"
    
    # Without system prompt
    print("\n Without system prompt:")
    print(f"Q: {question}")
    answer = llm.ask(question)
    print(f"A: {answer}\n")
    
    # With data analyst system prompt
    print("\n With data analyst system prompt:")
    print(f"Q: {question}")
    answer = llm.ask(question, system=templates.data_analyst_system())
    print(f"A: {answer}\n")
    
    # With concise assistant system prompt
    print("\n With concise assistant system prompt:")
    print(f"Q: {question}")
    answer = llm.ask(question, system=templates.concise_assistant_system())
    print(f"A: {answer}\n")


def test_structured_output():
    """Test getting JSON output"""
    print("\n" + "=" * 60)
    print("TEST 3: Structured JSON Output")
    print("=" * 60)
    
    llm = SimpleLLM()
    templates = PromptTemplates()
    
    prompt = templates.get_json_prompt(
        "Customer review: 'Amazing product! Fast shipping but packaging could be better. Worth the $99 price.'",
        ["overall_sentiment", "shipping_rating", "packaging_rating", "price_opinion"]
    )
    
    print(f"\n Prompt:\n{prompt}\n")
    
    answer = llm.ask(prompt)
    print(f" Response:\n{answer}\n")
    
    # Try to parse it
    import json
    try:
        parsed = json.loads(answer)
        print(" Valid JSON!")
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
    except:
        print(" Not valid JSON (might need prompt adjustment)")


def test_few_shot_learning():
    """Test few-shot classification"""
    print("\n" + "=" * 60)
    print("TEST 4: Few-Shot Classification")
    print("=" * 60)
    
    llm = SimpleLLM()
    templates = PromptTemplates()
    
    test_feedback = "The export feature doesn't work with Excel files"
    
    prompt = templates.few_shot_classification().format(text=test_feedback)
    
    print(f"\n Prompt:\n{prompt}\n")
    
    answer = llm.ask(prompt, system="You are a precise classifier. Return only the category name.")
    print(f" Classification: {answer}\n")


def test_data_analysis_prompt():
    """Test data analysis with real stats"""
    print("\n" + "=" * 60)
    print("TEST 5: Data Analysis Prompt")
    print("=" * 60)
    
    llm = SimpleLLM()
    templates = PromptTemplates()
    
    stats = {
        "total_reviews": 1000,
        "avg_rating": 4.2,
        "median_rating": 5.0,
        "std_rating": 1.1,
        "5_star": 650,
        "4_star": 200,
        "3_star": 80,
        "2_star": 40,
        "1_star": 30
    }
    
    prompt = templates.analyze_data_summary("Product reviews dataset", stats)
    
    print(f"\n Analyzing these stats:\n{prompt}\n")
    
    answer = llm.ask(prompt, system=templates.data_analyst_system())
    print(f" Analysis:\n{answer}\n")


def test_chain_of_thought():
    """Test chain-of-thought reasoning"""
    print("\n" + "=" * 60)
    print("TEST 6: Chain-of-Thought Reasoning")
    print("=" * 60)
    
    llm = SimpleLLM()
    templates = PromptTemplates()
    
    question = "Should we focus on improving 1-star reviews or increasing 5-star reviews?"
    
    prompt = templates.chain_of_thought_analysis(question)
    
    print(f"\n Question:\n{prompt}\n")
    
    answer = llm.ask(prompt, system=templates.data_analyst_system())
    print(f" Analysis:\n{answer}\n")


if __name__ == "__main__":
    print("\n PROMPT ENGINEERING EXPERIMENTS")
    print("=" * 60)
    print("Testing different prompt techniques with Groq LLM")
    print("=" * 60)
    
    # Run all tests
    try:
        test_temperature_effects()
        test_system_prompts()
        test_structured_output()
        test_few_shot_learning()
        test_data_analysis_prompt()
        test_chain_of_thought()
        
        print("\n" + "=" * 60)
        print(" ALL TESTS COMPLETE!")
        print("=" * 60)
        print("\nKey Learnings:")
        print("1. Lower temperature = more focused/consistent")
        print("2. System prompts significantly change output style")
        print("3. Few-shot examples improve accuracy")
        print("4. Structured output needs explicit format instructions")
        print("5. Chain-of-thought improves reasoning")
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check .env file has GROQ_API_KEY")
        print("2. Verify you're in the project root")
        print("3. Check imports are correct")