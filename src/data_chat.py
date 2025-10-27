from typing import Optional, Dict, Any
from data_loader import DataLoader
from conversation_manager import DataConversation
from text_generator import DataTextGenerator
import json


class DataChat:
    """
    Chat interface for interacting with datasets
    Combines data loading, conversation, and generation
    """
    
    def __init__(self, filepath: str):
        """
        Initialize data chat
        
        Args:
            filepath: Path to data file
        """
        print(" Loading data...")
        self.loader = DataLoader(filepath)
        
        print(" Initializing chat...")
        self.chat = DataConversation()
        
        print(" Setting up generators...")
        self.generator = DataTextGenerator()
        
        # Add dataset context to conversation
        self._setup_context()
        
        print("✓ Data chat ready!\n")
    
    def _setup_context(self):
        """Setup initial context with dataset information"""
        context_summary = self.loader.get_context_summary()
        
        # Add context to conversation
        self.chat.set_context(context_summary)
        
        # Generate initial summary
        initial_summary = self.generator.summarize_dataset(self.loader.metadata)
        
        print(" Dataset Summary:")
        print("-" * 60)
        print(initial_summary)
        print("-" * 60)
    
    def ask(self, question: str) -> str:
        """
        Ask a question about the dataset
        
        Args:
            question: User's question
            
        Returns:
            AI response
        """
        # Check if question is about specific column
        response = self._handle_special_queries(question)
        
        if response:
            return response
        
        # Otherwise, use conversation manager
        return self.chat.send_message(question)
    
    def _handle_special_queries(self, question: str) -> Optional[str]:
        """
        Handle special queries that need direct data access
        
        Args:
            question: User's question
            
        Returns:
            Response if special query, None otherwise
        """
        question_lower = question.lower()
        
        # Column-specific queries
        for col in self.loader.metadata['column_names']:
            col_lower = col.lower()
            
            # "tell me about [column]"
            if f"about {col_lower}" in question_lower or f"about the {col_lower}" in question_lower:
                col_info = self.loader.get_column_info(col)
                return self._format_column_response(col, col_info)
            
            # "what are the values in [column]"
            if f"values in {col_lower}" in question_lower or f"values of {col_lower}" in question_lower:
                counts = self.loader.get_value_counts(col, top_n=10)
                return self._format_value_counts_response(col, counts)
        
        # Correlation queries
        if "correlation" in question_lower and "between" in question_lower:
            return self._handle_correlation_query(question)
        
        return None
    
    def _format_column_response(self, col_name: str, col_info: Dict) -> str:
        """Format column information as response"""
        response = f"Here's what I found about the '{col_name}' column:\n\n"
        response += f"- Type: {col_info['dtype']}\n"
        response += f"- Total values: {col_info['count']}\n"
        response += f"- Missing: {col_info['missing']}\n"
        response += f"- Unique values: {col_info['unique']}\n"
        
        if 'statistics' in col_info:
            stats = col_info['statistics']
            response += f"\nStatistics:\n"
            response += f"- Mean: {stats['mean']:.2f}\n"
            response += f"- Median: {stats['median']:.2f}\n"
            response += f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
        
        if 'top_values' in col_info:
            response += f"\nTop values:\n"
            for val, count in list(col_info['top_values'].items())[:5]:
                response += f"- {val}: {count}\n"
        
        return response
    
    def _format_value_counts_response(self, col_name: str, counts: Dict) -> str:
        """Format value counts as response"""
        response = f"Here are the values in '{col_name}':\n\n"
        
        for val, count in counts.items():
            response += f"- {val}: {count} occurrences\n"
        
        total = sum(counts.values())
        response += f"\nTotal shown: {total}"
        
        return response
    
    def _handle_correlation_query(self, question: str) -> Optional[str]:
        """Handle correlation queries"""
        # Try to extract column names
        # Simple heuristic: look for column names in the question
        cols_found = []
        for col in self.loader.metadata['column_names']:
            if col.lower() in question.lower():
                cols_found.append(col)
        
        if len(cols_found) == 2:
            try:
                corr = self.loader.get_correlation(cols_found[0], cols_found[1])
                explanation = self.generator.explain_correlation(
                    cols_found[0],
                    cols_found[1],
                    corr
                )
                return f"Correlation between '{cols_found[0]}' and '{cols_found[1]}': {corr:.3f}\n\n{explanation}"
            except:
                pass
        
        return None
    
    def get_column_summary(self, column_name: str) -> str:
        """
        Get AI-generated summary of a column
        
        Args:
            column_name: Name of the column
            
        Returns:
            Summary text
        """
        col_info = self.loader.get_column_info(column_name)
        
        # Get sample values
        sample_values = col_info.get('sample_values', [])
        
        return self.generator.generate_column_description(
            column_name,
            sample_values
        )
    
    def get_insights(self, top_n: int = 3) -> list:
        """
        Generate insights about the dataset
        
        Args:
            top_n: Number of insights to generate
            
        Returns:
            List of insights
        """
        # Create data summary
        summary = self.loader.get_context_summary()
        
        # Generate insights
        return self.generator.generate_insights(summary, top_n)
    
    def compare_columns(self, col1: str, col2: str) -> str:
        """
        Compare two columns
        
        Args:
            col1: First column name
            col2: Second column name
            
        Returns:
            Comparison text
        """
        info1 = self.loader.get_column_info(col1)
        info2 = self.loader.get_column_info(col2)
        
        # Extract statistics if numeric
        stats1 = info1.get('statistics', {})
        stats2 = info2.get('statistics', {})
        
        if stats1 and stats2:
            return self.generator.compare_groups(col1, stats1, col2, stats2)
        else:
            return f"Comparison of '{col1}' and '{col2}':\n\n{json.dumps({'col1': info1, 'col2': info2}, indent=2)}"
    
    def generate_report(self) -> str:
        """
        Generate complete analysis report
        
        Returns:
            Report text
        """
        # Get insights
        insights = self.get_insights(top_n=5)
        
        # Get statistics summary
        stats_text = self.loader._format_statistics()
        
        # Generate report
        return self.generator.generate_analysis_report(
            self.loader.metadata['name'],
            insights,
            stats_text
        )
    
    def chat_history(self) -> list:
        """Get conversation history"""
        return self.chat.get_history()
    
    def save_conversation(self, filepath: str):
        """Save conversation"""
        self.chat.save_conversation(filepath)



if __name__ == "__main__":
    print(" Testing Data Chat\n")
    
    # Make sure sample data exists
    import pandas as pd
    from pathlib import Path
    
    Path('data').mkdir(exist_ok=True)
    
    sample_data = {
        'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse'],
        'price': [1200, 800, 600, 400, 150, 50],
        'rating': [4.5, 3.8, 4.2, 4.7, 3.5, 4.0],
        'reviews': [450, 1200, 890, 230, 180, 320],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_products.csv', index=False)
    
    print("=" * 60)
    print("TEST: Data Chat Integration")
    print("=" * 60)
    
    # Initialize
    data_chat = DataChat('data/sample_products.csv')
    
    # Test questions
    questions = [
        "How many products are in this dataset?",
        "What's the average price?",
        "Tell me about the rating column",
        "What are the values in category?",
        "What insights can you give me?"
    ]
    
    for q in questions:
        print(f"\n👤 User: {q}")
        answer = data_chat.ask(q)
        print(f"🤖 Assistant: {answer}\n")
        print("-" * 60)
    
    print("\n Data chat test complete!")