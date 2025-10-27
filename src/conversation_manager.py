import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

# sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import SimpleLLM


class ConversationManager:
    """
    Manage multi-turn conversations with context
    """
    
    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize conversation manager
        
        Args:
            system_message: Optional system instructions for the AI
        """
        self.llm = SimpleLLM()
        self.system_message = system_message or "You are a helpful data analysis assistant."
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 20  # Keep last 20 messages
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add system message to history
        self.conversation_history.append({
            "role": "system",
            "content": self.system_message
        })
        
        print(f"✓ Conversation started (ID: {self.conversation_id})")
    
    def send_message(self, user_message: str) -> str:
        """
        Send a message and get response
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response from LLM
        response = self.llm.chat(self.conversation_history)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Trim history if too long (keep system message + last N messages)
        if len(self.conversation_history) > self.max_history + 1:
            # Keep system message (index 0) and recent messages
            self.conversation_history = (
                [self.conversation_history[0]] +  # System message
                self.conversation_history[-(self.max_history):]  # Recent messages
            )
        
        return response
    
    def get_history(self, include_system: bool = False) -> List[Dict[str, str]]:
        """
        Get conversation history
        
        Args:
            include_system: Whether to include system message
            
        Returns:
            List of messages
        """
        if include_system:
            return self.conversation_history
        else:
            # Skip system message (index 0)
            return self.conversation_history[1:]
    
    def clear_history(self, keep_system: bool = True):
        """
        Clear conversation history
        
        Args:
            keep_system: Whether to keep system message
        """
        if keep_system:
            # Keep only system message
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []
        
        print("✓ Conversation history cleared")
    
    def save_conversation(self, filepath: str):
        """
        Save conversation to file
        
        Args:
            filepath: Path to save conversation
        """
        conversation_data = {
            "conversation_id": self.conversation_id,
            "system_message": self.system_message,
            "messages": self.get_history(include_system=False),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"✓ Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """
        Load conversation from file
        
        Args:
            filepath: Path to load conversation from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.conversation_id = data.get("conversation_id", self.conversation_id)
        self.system_message = data.get("system_message", self.system_message)
        
        # Rebuild history
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]
        self.conversation_history.extend(data.get("messages", []))
        
        print(f"✓ Conversation loaded from {filepath}")
    
    def get_message_count(self) -> int:
        """Get number of messages (excluding system)"""
        return len(self.conversation_history) - 1
    
    def set_context(self, context: str):
        """
        Add context to system message
        
        Args:
            context: Additional context (e.g., data summary)
        """
        # Update system message with context
        new_system = f"{self.system_message}\n\nContext:\n{context}"
        self.conversation_history[0]["content"] = new_system
        print("✓ Context added to conversation")


class DataConversation(ConversationManager):
    """
    Specialized conversation manager for data analysis
    """
    
    def __init__(self, dataset_info: Optional[Dict] = None):
        """
        Initialize data conversation
        
        Args:
            dataset_info: Information about the dataset
        """
        system_message = """You are an expert data analyst assistant.

            Your role:
            - Answer questions about the dataset
            - Provide insights and recommendations
            - Explain statistical concepts clearly
            - Suggest analysis approaches
            - Help interpret results

            Style:
            - Be concise and clear
            - Use specific numbers from the data
            - Provide actionable insights
            - Ask clarifying questions when needed"""
        
        super().__init__(system_message)
        
        # Add dataset context if provided
        if dataset_info:
            self.add_dataset_context(dataset_info)
    
    def add_dataset_context(self, dataset_info: Dict):
        """
        Add dataset information to context
        
        Args:
            dataset_info: Dataset metadata and statistics
        """
        context = f"""
            Dataset Information:
            - Name: {dataset_info.get('name', 'Unknown')}
            - Rows: {dataset_info.get('rows', 'N/A')}
            - Columns: {dataset_info.get('columns', 'N/A')}
            - Column Names: {', '.join(dataset_info.get('column_names', []))}

            Statistics:
            {json.dumps(dataset_info.get('statistics', {}), indent=2)}

            The user will ask questions about this dataset. Use this context to provide specific, accurate answers.
        """
        self.set_context(context)
        print("✓ Dataset context loaded")
    
    def ask_about_column(self, column_name: str, question: str) -> str:
        """
        Ask a question about a specific column
        
        Args:
            column_name: Name of the column
            question: Question about the column
            
        Returns:
            Assistant's response
        """
        full_question = f"Regarding the '{column_name}' column: {question}"
        return self.send_message(full_question)
    
    def get_recommendation(self, analysis_type: str) -> str:
        """
        Get recommendation for analysis
        
        Args:
            analysis_type: Type of analysis (e.g., 'correlation', 'trend', 'outliers')
            
        Returns:
            Recommendation
        """
        question = f"What {analysis_type} analysis would you recommend for this dataset and why?"
        return self.send_message(question)


# Quick test
if __name__ == "__main__":
    print("🧪 Testing Conversation Manager\n")
    
    # Test 1: Basic conversation
    print("=" * 60)
    print("TEST 1: Basic Multi-turn Conversation")
    print("=" * 60)
    
    chat = ConversationManager()
    
    print("\n User: What is machine learning?")
    response = chat.send_message("What is machine learning?")
    print(f" Assistant: {response}\n")
    
    print(" User: Give me an example")
    response = chat.send_message("Give me an example")
    print(f" Assistant: {response}\n")
    
    print(" User: How does that relate to what you just said?")
    response = chat.send_message("How does that relate to what you just said?")
    print(f" Assistant: {response}\n")
    
    print(f" Total messages: {chat.get_message_count()}")
    
    # Test 2: Data conversation
    print("\n" + "=" * 60)
    print("TEST 2: Data-Aware Conversation")
    print("=" * 60)
    
    dataset_info = {
        'name': 'Customer Reviews',
        'rows': 1000,
        'columns': 5,
        'column_names': ['product_id', 'rating', 'review_text', 'date', 'verified_purchase'],
        'statistics': {
            'rating': {'mean': 4.2, 'median': 4.5, 'std': 0.8},
            'verified_purchase': {'true': 850, 'false': 150}
        }
    }
    
    data_chat = DataConversation(dataset_info)
    
    print("\n👤 User: What can you tell me about this dataset?")
    response = data_chat.send_message("What can you tell me about this dataset?")
    print(f" Assistant: {response}\n")
    
    print("👤 User: What's the average rating?")
    response = data_chat.send_message("What's the average rating?")
    print(f" Assistant: {response}\n")
    
    print("👤 User: Is that good?")
    response = data_chat.send_message("Is that good?")
    print(f" Assistant: {response}\n")
    
    # Test 3: Save/Load
    print("\n" + "=" * 60)
    print("TEST 3: Save and Load Conversation")
    print("=" * 60)
    
    chat.save_conversation("output/test_conversation.json")
    
    new_chat = ConversationManager()
    new_chat.load_conversation("output/test_conversation.json")
    
    print(f"✓ Loaded conversation with {new_chat.get_message_count()} messages")
    
    print("\n" + "=" * 60)
    print(" All tests complete!")
    print("=" * 60)