# 🧠 Prompt Engineering Guide

## What is Prompt Engineering?

The art and science of crafting inputs to get the best outputs from LLMs.

## Key Concepts

### 1. Temperature (Randomness)
```python
temperature=0.0  # Deterministic, focused, consistent
temperature=0.3  # Slightly creative, mostly focused
temperature=0.7  # Balanced (default)
temperature=1.0  # Creative, varied, unpredictable
temperature=2.0  # Very random (rarely useful)
```

**Use cases:**
- `0.0-0.3` → Data analysis, factual answers, code generation
- `0.5-0.8` → Creative writing, brainstorming
- `0.9-1.5` → Poetry, experimental outputs

### 2. System Messages

Instructions that set the AI's behavior for the entire conversation.

**Good system prompts:**
```
"You are a data analyst. Provide concise, actionable insights."
"You are a Python expert. Write clean, well-documented code."
"You are a helpful assistant that answers in bullet points."
```

**Bad system prompts:**
```
"Be helpful."  # Too vague
"You are the best AI ever!"  # Not useful
```

### 3. Few-Shot Learning

Show the AI examples of what you want.

**Structure:**
```
Example 1: Input → Output
Example 2: Input → Output
Example 3: Input → Output
Now: Your actual input
```

### 4. Chain-of-Thought

Ask the AI to think step-by-step.

**Magic phrase:** "Let's think step by step:"

### 5. Structured Output

Ask for specific formats: JSON, CSV, tables, lists.

## Best Practices

✅ **DO:**
- Be specific and clear
- Provide examples
- Set clear expectations
- Use appropriate temperature
- Request specific formats

❌ **DON'T:**
- Be vague ("help me")
- Use emotional language
- Assume AI knows context
- Forget to specify output format