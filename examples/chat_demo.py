import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation_manager import ConversationManager, DataConversation
from src.prompt_templates import PromptTemplates
import time


def simulate_conversation(chat, conversations):
    """Simulate a conversation with delays"""
    for user_msg, assistant_label in conversations:
        print(f"\n👤 User: {user_msg}")
        time.sleep(0.5)
        
        print(f"🤖 {assistant_label}: ", end="", flush=True)
        response = chat.send_message(user_msg)
        print(response)
        time.sleep(1)


def demo_context_memory():
    """Demonstrate context memory"""
    print("\n" + "=" * 70)
    print("DEMO 1: Context Memory - Following the Conversation")
    print("=" * 70)
    
    chat = ConversationManager()
    
    conversations = [
        ("What is correlation?", "Assistant"),
        ("Give me a simple example", "Assistant"),  # References "correlation"
        ("How is that different from causation?", "Assistant"),  # References "that"
        ("Which one is more important?", "Assistant"),  # References both concepts
    ]
    
    simulate_conversation(chat, conversations)
    
    print(f"\n📊 Conversation had {chat.get_message_count()} messages")


def demo_data_context():
    """Demonstrate data-aware conversation"""
    print("\n" + "=" * 70)
    print("DEMO 2: Data-Aware Conversation")
    print("=" * 70)
    
    dataset_info = {
        'name': 'Sales Data Q4 2024',
        'rows': 5000,
        'columns': 8,
        'column_names': ['date', 'product', 'quantity', 'price', 'region', 'customer_type', 'discount', 'revenue'],
        'statistics': {
            'revenue': {'mean': 245.50, 'median': 198.00, 'std': 125.30},
            'discount': {'mean': 0.15, 'min': 0, 'max': 0.50},
            'quantity': {'mean': 3.2, 'median': 2.0}
        }
    }
    
    chat = DataConversation(dataset_info)
    
    conversations = [
        ("Summarize this dataset for me", "Assistant"),
        ("What's the average revenue per order?", "Assistant"),
        ("Is that above or below typical e-commerce averages?", "Assistant"),
        ("What analysis would you recommend?", "Assistant"),
    ]
    
    simulate_conversation(chat, conversations)


def demo_save_load():
    """Demonstrate save/load functionality"""
    print("\n" + "=" * 70)
    print("DEMO 3: Save and Load Conversations")
    print("=" * 70)
    
    # Create conversation
    print("\n📝 Creating conversation...")
    chat1 = ConversationManager()
    chat1.send_message("What is data cleaning?")
    chat1.send_message("What are the main steps?")
    
    # Save
    print("\n💾 Saving conversation...")
    chat1.save_conversation("output/demo_conversation.json")
    
    # Load in new instance
    print("\n📂 Loading in new conversation instance...")
    chat2 = ConversationManager()
    chat2.load_conversation("output/demo_conversation.json")
    
    print(f"✓ Loaded {chat2.get_message_count()} messages")
    
    # Continue conversation
    print("\n▶️  Continuing loaded conversation...")
    print("\n👤 User: Can you remind me what we were discussing?")
    response = chat2.send_message("Can you remind me what we were discussing?")
    print(f"🤖 Assistant: {response}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎬 CONVERSATION MANAGER DEMONSTRATIONS")
    print("=" * 70)
    
    demo_context_memory()
    demo_data_context()
    demo_save_load()
    
    print("\n" + "=" * 70)
    print("✅ ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\n💡 Key Features Demonstrated:")
    print("  ✓ Multi-turn conversations with context")
    print("  ✓ Data-aware conversations")
    print("  ✓ Save/load conversation state")
    print("  ✓ Context memory across messages")