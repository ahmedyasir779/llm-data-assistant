import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_chat import DataChat
import time


def simulate_chat(data_chat, questions):
    """Simulate a conversation"""
    for question in questions:
        print(f"\n👤 User: {question}")
        time.sleep(0.5)
        
        print("🤖 Assistant: ", end="", flush=True)
        response = data_chat.ask(question)
        print(response)
        time.sleep(1)


def demo_product_analysis():
    """Demo with products dataset"""
    print("\n" + "=" * 70)
    print("DEMO 1: Product Dataset Analysis")
    print("=" * 70)
    
    data_chat = DataChat('data/products.csv')
    
    questions = [
        "How many products are in this dataset?",
        "What's the price range?",
        "What categories exist?",
        "What's the highest rated product?",
        "What's the correlation between price and rating?",
        "Which category has the most products?",
    ]
    
    simulate_chat(data_chat, questions)
    
    # Show insights
    print("\n" + "-" * 70)
    print("💡 GENERATING INSIGHTS...")
    print("-" * 70)
    insights = data_chat.get_insights(top_n=5)
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")


def demo_review_analysis():
    """Demo with reviews dataset"""
    print("\n" + "=" * 70)
    print("DEMO 2: Customer Reviews Analysis")
    print("=" * 70)
    
    data_chat = DataChat('data/reviews.csv')
    
    questions = [
        "What's the average rating?",
        "How many reviews are verified purchases?",
        "What percentage of reviews are 5 stars?",
        "Tell me about the helpful_votes column",
    ]
    
    simulate_chat(data_chat, questions)


def demo_sales_analysis():
    """Demo with sales dataset"""
    print("\n" + "=" * 70)
    print("DEMO 3: Sales Data Analysis")
    print("=" * 70)
    
    data_chat = DataChat('data/sales.csv')
    
    questions = [
        "How many sales records are there?",
        "What customer types do we have?",
        "What are the values in region?",
        "What's the average discount given?",
        "What insights can you provide about this sales data?",
    ]
    
    simulate_chat(data_chat, questions)
    
    # Generate report
    print("\n" + "-" * 70)
    print("📄 GENERATING FULL REPORT...")
    print("-" * 70)
    report = data_chat.generate_report()
    print(report)


def demo_column_exploration():
    """Demo detailed column exploration"""
    print("\n" + "=" * 70)
    print("DEMO 4: Column-Level Exploration")
    print("=" * 70)
    
    data_chat = DataChat('data/products.csv')
    
    print("\n🔍 Exploring 'price' column...")
    summary = data_chat.get_column_summary('price')
    print(summary)
    
    print("\n🔍 Exploring 'rating' column...")
    summary = data_chat.get_column_summary('rating')
    print(summary)
    
    print("\n⚖️ Comparing price and rating...")
    comparison = data_chat.compare_columns('price', 'rating')
    print(comparison)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎬 DATA CHAT DEMONSTRATIONS")
    print("=" * 70)
    
    # Make sure sample data exists
    if not Path('data/products.csv').exists():
        print("\n⚠️  Sample data not found. Running create_sample_data.py first...")
        import subprocess
        subprocess.run([sys.executable, 'create_sample_data.py'])
        print()
    
    # Run all demos
    demo_product_analysis()
    demo_review_analysis()
    demo_sales_analysis()
    demo_column_exploration()
    
    print("\n" + "=" * 70)
    print("✅ ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 70)
    print("\n💡 Key Features Demonstrated:")
    print("  ✓ Load and analyze CSV files")
    print("  ✓ Natural language questions")
    print("  ✓ Automatic data context")
    print("  ✓ Column-specific queries")
    print("  ✓ Correlation analysis")
    print("  ✓ Insight generation")
    print("  ✓ Report generation")
    print("  ✓ Multi-turn conversations")