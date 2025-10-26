# 💬 Conversation Memory Guide

## How LLMs Handle Context

LLMs are **stateless** - they don't remember past conversations unless you provide the history.

## Conversation Structure
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "What about 3+3?"},
    {"role": "assistant", "content": "3+3 equals 6."},
    {"role": "user", "content": "Add those two results"}  # ← Needs previous context!
]
```

## Message Roles

- **system**: Instructions for the AI's behavior
- **user**: Human's messages
- **assistant**: AI's responses

## Memory Management

### Simple Approach 
```python
conversation_history = []

# Add user message
conversation_history.append({"role": "user", "content": question})

# Get response
response = llm.chat(conversation_history)

# Add assistant response
conversation_history.append({"role": "assistant", "content": response})
```

### Token Limits
- Models have max context length (8k, 32k, 128k tokens)
- ~1 token = 4 characters
- Need to trim old messages when limit approached

## Best Practices

✅ **DO:**
- Keep system message consistent
- Store full conversation history
- Trim old messages when near limit
- Provide relevant context

❌ **DON'T:**
- Forget to add assistant responses to history
- Let conversation grow indefinitely
- Change system message mid-conversation