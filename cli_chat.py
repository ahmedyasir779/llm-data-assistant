import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.conversation_manager import ConversationManager, DataConversation
from src.prompt_templates import PromptTemplates


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print(" LLM DATA ASSISTANT - INTERACTIVE CHAT")
    print("=" * 70)
    print("Chat with AI about data analysis")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 70 + "\n")


def print_help():
    """Print help message"""
    help_text = """
        AVAILABLE COMMANDS:

        Basic Commands:
        help          - Show this help message
        quit/exit     - Exit the chat
        clear         - Clear conversation history
        history       - Show conversation history
        save          - Save conversation to file
        load          - Load conversation from file
        
        Chat Commands:
        <message>     - Send a message to the AI
        
        Examples:
        What is machine learning?
        Explain correlation in simple terms
        How do I clean missing data?
        
        Tips:
        - Ask follow-up questions (context is remembered)
        - Be specific for better answers
        - Reference previous messages ("that", "it", etc.)
    """
    print(help_text)


def display_history(chat: ConversationManager):
    """Display conversation history"""
    history = chat.get_history(include_system=False)
    
    if not history:
        print(" No messages yet. Start chatting!\n")
        return
    
    print("\n" + "=" * 70)
    print(" CONVERSATION HISTORY")
    print("=" * 70)
    
    for i, msg in enumerate(history, 1):
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            print(f"\n[{i}]  You:")
            print(f"    {content}")
        else:
            print(f"\n[{i}]  Assistant:")
            print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")
    
    print("\n" + "=" * 70)
    print(f"Total messages: {len(history)}")
    print("=" * 70 + "\n")


def save_conversation(chat: ConversationManager):
    """Save conversation with user prompt"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"output/conversation_{timestamp}.json"
    
    print(f"\n Save conversation")
    filename = input(f"Filename [{default_filename}]: ").strip()
    
    if not filename:
        filename = default_filename
    
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    try:
        chat.save_conversation(filename)
        print(f" Saved to {filename}\n")
    except Exception as e:
        print(f" Error saving: {e}\n")


def load_conversation(chat: ConversationManager):
    """Load conversation with user prompt"""
    print("\n Load conversation")
    filename = input("Filename: ").strip()
    
    if not filename:
        print(" No filename provided\n")
        return
    
    try:
        chat.load_conversation(filename)
        print(f"✓ Loaded from {filename}")
        print(f"✓ {chat.get_message_count()} messages restored\n")
    except Exception as e:
        print(f" Error loading: {e}\n")


def chat_session_basic():
    """Run basic chat session"""
    print_banner()
    
    # Initialize conversation
    chat = ConversationManager(
        system_message=PromptTemplates.data_analyst_system()
    )
    
    print(" Assistant: Hi! I'm your data analysis assistant. Ask me anything!\n")
    
    while True:
        try:
            # Get user input
            user_input = input(" You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye! Have a great day!\n")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'clear':
                chat.clear_history()
                print("✓ History cleared\n")
                continue
            
            elif user_input.lower() == 'history':
                display_history(chat)
                continue
            
            elif user_input.lower() == 'save':
                save_conversation(chat)
                continue
            
            elif user_input.lower() == 'load':
                load_conversation(chat)
                continue
            
            # Send message and get response
            print(" Assistant: ", end="", flush=True)
            response = chat.send_message(user_input)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\n\n Interrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n Error: {e}\n")


def chat_session_with_data():
    """Run chat session with dataset context"""
    print_banner()
    print(" DATA-AWARE CHAT MODE\n")
    
    # Example dataset info
    dataset_info = {
        'name': 'Customer Reviews Dataset',
        'rows': 1000,
        'columns': 5,
        'column_names': ['product_id', 'rating', 'review_text', 'date', 'verified_purchase'],
        'statistics': {
            'rating': {
                'mean': 4.2,
                'median': 4.5,
                'std': 0.8,
                'min': 1.0,
                'max': 5.0
            },
            'verified_purchase': {
                'true': 850,
                'false': 150
            }
        }
    }
    
    # Initialize data conversation
    chat = DataConversation(dataset_info)
    
    print(" Assistant: I've loaded the Customer Reviews dataset.")
    print(" Assistant: It has 1,000 reviews across 5 columns.")
    print(" Assistant: Ask me anything about it!\n")
    
    while True:
        try:
            # Get user input
            user_input = input(" You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!\n")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                print("\n DATA COMMANDS:")
                print("  What columns are in the dataset?")
                print("  What's the average rating?")
                print("  Tell me about verified purchases")
                print("  What analysis would you recommend?\n")
                continue
            
            elif user_input.lower() == 'clear':
                chat.clear_history()
                # Re-add dataset context
                chat.add_dataset_context(dataset_info)
                print("✓ History cleared (dataset context preserved)\n")
                continue
            
            elif user_input.lower() == 'history':
                display_history(chat)
                continue
            
            elif user_input.lower() == 'save':
                save_conversation(chat)
                continue
            
            # Send message and get response
            print(" Assistant: ", end="", flush=True)
            response = chat.send_message(user_input)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\n\n Interrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n Error: {e}\n")


def main():
    """Main entry point"""
    print("\n Choose chat mode:\n")
    print("1. Basic Chat (General data analysis questions)")
    print("2. Data-Aware Chat (Chat about specific dataset)")
    print("3. Exit\n")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        chat_session_basic()
    elif choice == '2':
        chat_session_with_data()
    elif choice == '3':
        print("\n Goodbye!\n")
    else:
        print("\n Invalid choice\n")


if __name__ == "__main__":
    main()