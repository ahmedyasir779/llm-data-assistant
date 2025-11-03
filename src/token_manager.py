import tiktoken
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class TokenManager:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 4096
    ):
        """
        Initialize token manager
        
        Args:
            model: Model name for tokenizer
            max_tokens: Maximum tokens allowed
        """
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for most models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.model = model
        self.max_tokens = max_tokens
        
        # Token allocation strategy
        self.allocation = {
            "system_prompt": 0.10,      # 10% for system instructions
            "context": 0.50,             # 50% for retrieved context
            "sample_data": 0.30,         # 30% for sample data
            "question": 0.05,            # 5% for question
            "buffer": 0.05               # 5% safety buffer
        }
        
        print(f"🎫 Token manager initialized")
        print(f"   Model: {model}")
        print(f"   Max tokens: {max_tokens:,}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Text to count
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]
    
    def get_budget(self, component: str) -> int:
        """
        Get token budget for component
        
        Args:
            component: Component name (system_prompt, context, etc)
            
        Returns:
            Token budget
        """
        allocation = self.allocation.get(component, 0.0)
        return int(self.max_tokens * allocation)
    
    def truncate_to_budget(
        self,
        text: str,
        budget: int,
        truncate_from: str = "end"
    ) -> str:
        """
        Truncate text to fit token budget
        
        Args:
            text: Text to truncate
            budget: Token budget
            truncate_from: 'start', 'end', or 'middle'
            
        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= budget:
            return text
        
        if truncate_from == "end":
            # Keep beginning
            truncated_tokens = tokens[:budget]
        elif truncate_from == "start":
            # Keep end
            truncated_tokens = tokens[-budget:]
        else:  # middle
            # Keep beginning and end
            keep_each = budget // 2
            truncated_tokens = tokens[:keep_each] + tokens[-keep_each:]
        
        return self.encoding.decode(truncated_tokens)
    
    def optimize_context_allocation(
        self,
        system_prompt: str,
        context: str,
        sample_data: str,
        question: str
    ) -> Dict[str, str]:
        """
        Optimize allocation of tokens across components
        
        Args:
            system_prompt: System prompt text
            context: Retrieved context
            sample_data: Sample data
            question: User question
            
        Returns:
            Optimized texts within budget
        """
        # Count current tokens
        current_tokens = {
            "system_prompt": self.count_tokens(system_prompt),
            "context": self.count_tokens(context),
            "sample_data": self.count_tokens(sample_data),
            "question": self.count_tokens(question)
        }
        
        total_current = sum(current_tokens.values())
        
        # If within budget, return as-is
        if total_current <= self.max_tokens * 0.95:  # 95% to be safe
            return {
                "system_prompt": system_prompt,
                "context": context,
                "sample_data": sample_data,
                "question": question,
                "stats": {
                    "total_tokens": total_current,
                    "budget_used": f"{total_current/self.max_tokens*100:.1f}%",
                    "truncated": False
                }
            }
        
        # Need to truncate - allocate budgets
        budgets = {
            component: self.get_budget(component)
            for component in ["system_prompt", "context", "sample_data", "question"]
        }
        
        # Truncate each component to budget
        optimized = {
            "system_prompt": self.truncate_to_budget(
                system_prompt, budgets["system_prompt"], "end"
            ),
            "context": self.truncate_to_budget(
                context, budgets["context"], "middle"
            ),
            "sample_data": self.truncate_to_budget(
                sample_data, budgets["sample_data"], "end"
            ),
            "question": question  # Never truncate question
        }
        
        # Calculate final stats
        final_tokens = sum(self.count_tokens(v) for v in optimized.values())
        
        optimized["stats"] = {
            "total_tokens": final_tokens,
            "budget_used": f"{final_tokens/self.max_tokens*100:.1f}%",
            "truncated": True,
            "original_tokens": total_current,
            "saved_tokens": total_current - final_tokens
        }
        
        return optimized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token manager statistics"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "allocation": self.allocation,
            "budgets": {
                component: self.get_budget(component)
                for component in self.allocation.keys()
            }
        }


class ContextWindow:
    """
    Manages sliding context window for long conversations
    """
    
    def __init__(
        self,
        max_messages: int = 10,
        max_tokens: int = 4096,
        token_manager: Optional[TokenManager] = None
    ):
        """
        Initialize context window
        
        Args:
            max_messages: Maximum number of messages to keep
            max_tokens: Maximum total tokens
            token_manager: Token manager instance
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.token_manager = token_manager or TokenManager(max_tokens=max_tokens)
        
        print(f"🪟 Context window initialized")
        print(f"   Max messages: {max_messages}")
        print(f"   Max tokens: {max_tokens:,}")
    
    def manage_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Manage message history to fit within window
        
        Args:
            messages: List of message dicts
            
        Returns:
            Filtered messages within limits
        """
        if not messages:
            return messages
        
        # Always keep system message
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]
        
        # Apply message count limit
        if len(other_messages) > self.max_messages:
            # Keep most recent messages
            other_messages = other_messages[-self.max_messages:]
        
        # Check token limit
        all_messages = system_messages + other_messages
        total_tokens = sum(
            self.token_manager.count_tokens(m.get("content", ""))
            for m in all_messages
        )
        
        # If over token limit, remove oldest messages
        while total_tokens > self.max_tokens and len(other_messages) > 1:
            removed = other_messages.pop(0)
            removed_tokens = self.token_manager.count_tokens(
                removed.get("content", "")
            )
            total_tokens -= removed_tokens
        
        return system_messages + other_messages
    
    def summarize_old_context(
        self,
        messages: List[Dict[str, str]],
        keep_recent: int = 3
    ) -> List[Dict[str, str]]:
        """
        Summarize old messages to save tokens
        
        Args:
            messages: Message list
            keep_recent: Number of recent messages to keep as-is
            
        Returns:
            Messages with summarized history
        """
        if len(messages) <= keep_recent + 1:  # +1 for system
            return messages
        
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]
        
        # Keep recent messages
        recent = other_messages[-keep_recent:]
        old = other_messages[:-keep_recent]
        
        # Create summary of old messages
        if old:
            summary_text = "Previous conversation summary:\n"
            for msg in old:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:100]  # First 100 chars
                summary_text += f"- {role}: {content}...\n"
            
            summary_message = {
                "role": "system",
                "content": summary_text
            }
            
            return system_messages + [summary_message] + recent
        
        return system_messages + recent