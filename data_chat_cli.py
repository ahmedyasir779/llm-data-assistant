import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_chat import DataChat


def print_banner(filename: str):
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print(" DATA CHAT - Talk to Your Data")
    print("=" * 70)
    print(f"File: {filename}")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 70 + "\n")


def print_help():
    """Print help message"""
    help_text = """
        AVAILABLE COMMANDS:

        Basic Commands:
        help          - Show this help message
        quit/exit     - Exit the chat
        history       - Show conversation history
        save          - Save conversation to file
        
        Data Commands:
        insights      - Generate key insights about the data
        report        - Generate full analysis report
        columns       - List all columns
        summary       - Show data summary again
        
        Chat Naturally:
        How many rows are in this dataset?
        What's the average price?
        Tell me about the rating column
        What are the top categories?
        What's the correlation between price and rating?
        Compare price and reviews
        What insights do you see?
        
        Tips:
        - Ask follow-up questions (context is remembered)
        - Reference specific columns by name
        - Ask for explanations ("why is that?")
        - Request recommendations ("what should I analyze?")
        """
    print(help_text)


def display_history(data_chat: DataChat):
    """Display conversation history"""
    history = data_chat.chat_history()
    
    if not history:
        print(" No messages yet.\n")
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
            # Truncate long responses
            if len(content) > 300:
                print(f"    {content[:300]}...")
            else:
                print(f"    {content}")
    
    print("\n" + "=" * 70 + "\n")


def show_columns(data_chat: DataChat):
    """Show all columns"""
    print("\n COLUMNS:")
    print("-" * 60)
    for i, col in enumerate(data_chat.loader.metadata['column_names'], 1):
        dtype = data_chat.loader.metadata['dtypes'][col]
        print(f"{i}. {col} ({dtype})")
    print("-" * 60 + "\n")


def show_insights(data_chat: DataChat):
    """Generate and show insights"""
    print("\n Generating insights...\n")
    insights = data_chat.get_insights(top_n=5)
    
    print("=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    for insight in insights:
        print(insight)
    print("=" * 70 + "\n")


def show_report(data_chat: DataChat):
    """Generate and show full report"""
    print("\n Generating full report...\n")
    report = data_chat.generate_report()
    
    print("=" * 70)
    print("ANALYSIS REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70 + "\n")


def save_conversation(data_chat: DataChat):
    """Save conversation"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"output/data_chat_{timestamp}.json"
    
    print(f"\n Save conversation")
    filename = input(f"Filename [{default_filename}]: ").strip()
    
    if not filename:
        filename = default_filename
    
    Path("output").mkdir(exist_ok=True)
    
    try:
        data_chat.save_conversation(filename)
        print(f"✓ Saved to {filename}\n")
    except Exception as e:
        print(f"❌ Error saving: {e}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Chat with your data files')
    parser.add_argument('file', help='Path to data file (CSV, Excel, JSON)')
    args = parser.parse_args()
    
    filepath = args.file
    
    if not Path(filepath).exists():
        print(f"❌ Error: File not found: {filepath}")
        return
    
    try:
        # Initialize data chat
        data_chat = DataChat(filepath)
        
        print_banner(Path(filepath).name)
        
        # Main chat loop
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
                    continue
                
                elif user_input.lower() == 'history':
                    display_history(data_chat)
                    continue
                
                elif user_input.lower() == 'columns':
                    show_columns(data_chat)
                    continue
                
                elif user_input.lower() == 'summary':
                    print("\n" + data_chat.loader.get_context_summary())
                    continue
                
                elif user_input.lower() == 'insights':
                    show_insights(data_chat)
                    continue
                
                elif user_input.lower() == 'report':
                    show_report(data_chat)
                    continue
                
                elif user_input.lower() == 'save':
                    save_conversation(data_chat)
                    continue
                
                # Send message to AI
                print(" Assistant: ", end="", flush=True)
                response = data_chat.ask(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\n Interrupted. Type 'quit' to exit.\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    except Exception as e:
        print(f"\n❌ Error loading data: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n DATA CHAT")
        print("\nUsage: python data_chat_cli.py <data_file>")
        print("\nExamples:")
        print("  python data_chat_cli.py data/sales.csv")
        print("  python data_chat_cli.py data/customers.xlsx")
        print("\n")
    else:
        main()