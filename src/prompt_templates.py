from typing import List, Dict, Any
import json

class PromptTemplates:
    @staticmethod
    def data_analyst_system() -> str:
        """System prompt for data analysis tasks"""
        return """You are an expert data analyst with 10+ years of experience.
                  Your role:
                    - Analyze data clearly and concisely
                    - Identify patterns and insights
                    - Provide actionable recommendations
                    - Use statistical reasoning
                    - Explain findings in simple terms

                    Style:
                    - Be direct and precise
                    - Use bullet points for clarity
                    - Include specific numbers
                    - Avoid jargon unless necessary"""
    
    @staticmethod
    def python_expert_system() -> str:
        """System prompt for code generation"""
        return """You are a senior Python developer specializing in data science.

                  Your code:
                    - Is clean and well-documented
                    - Follows PEP 8 style
                    - Includes error handling
                    - Has clear variable names
                    - Contains helpful comments

                    Always explain your code briefly."""
    
    @staticmethod
    def concise_assistant_system() -> str:
        """System prompt for short answers"""
        return """You are a helpful assistant that provides concise, accurate answers.

                    Rules:
                    - Keep answers under 3 sentences when possible
                    - Be direct and specific
                    - No unnecessary elaboration
                    - Focus on the key point"""
    
    # ═══════════════════════════════════════════════════════════
    # DATA ANALYSIS PROMPTS
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def analyze_data_summary(data_description: str, stats: Dict) -> str:
        """
        Prompt for analyzing data summary statistics
        
        Args:
            data_description: What the data represents
            stats: Dictionary of statistics
        """
        return f"""Analyze this dataset and provide key insights:

                Dataset: {data_description}

                Statistics:
                {json.dumps(stats, indent=2)}

                Please provide:
                1. Key observations (3-5 bullet points)
                2. Notable patterns or outliers
                3. One actionable recommendation

                Keep it concise and business-focused."""
    
    @staticmethod
    def compare_columns(col1_name: str, col1_stats: Dict, 
                       col2_name: str, col2_stats: Dict) -> str:
        """Prompt for comparing two columns"""
        return f"""Compare these two data columns and identify the relationship:

                Column 1: {col1_name}
                {json.dumps(col1_stats, indent=2)}

                Column 2: {col2_name}
                {json.dumps(col2_stats, indent=2)}

                Analysis needed:
                1. What's the likely relationship?
                2. Are they correlated?
                3. What does this tell us about the data?

                Be specific and use the numbers provided."""
    
    @staticmethod
    def sentiment_summary(text_data: List[str], sentiment_scores: List[float]) -> str:
        """Prompt for sentiment analysis summary"""
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        positive = sum(1 for s in sentiment_scores if s > 0.1)
        negative = sum(1 for s in sentiment_scores if s < -0.1)
        neutral = len(sentiment_scores) - positive - negative
        
        return f"""Analyze this sentiment data and provide insights:

                Total entries: {len(text_data)}
                Average sentiment: {avg_sentiment:.3f}
                Positive: {positive} ({positive/len(text_data)*100:.1f}%)
                Neutral: {neutral} ({neutral/len(text_data)*100:.1f}%)
                Negative: {negative} ({negative/len(text_data)*100:.1f}%)

                Sample texts:
                {chr(10).join([f'- "{text[:100]}..."' for text in text_data[:3]])}

                Provide:
                1. Overall sentiment assessment
                2. Key themes (if obvious from samples)
                3. Actionable insight for stakeholders

                Keep it business-focused."""
    
    # ═══════════════════════════════════════════════════════════
    # FEW-SHOT EXAMPLES
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def few_shot_classification() -> str:
        """Few-shot learning for text classification"""
        return """Classify the following customer feedback into categories: Bug, Feature Request, Compliment, or Complaint.

        Examples:
        Feedback: "The app crashes when I try to export data"
        Category: Bug

        Feedback: "Would love to see dark mode added"
        Category: Feature Request

        Feedback: "Best analytics tool I've ever used!"
        Category: Compliment

        Feedback: "Customer support took 3 days to respond"
        Category: Complaint

        Now classify this:
        Feedback: "{text}"
        Category:"""
    
    @staticmethod
    def few_shot_extraction() -> str:
        """Few-shot for extracting key info"""
        return """Extract the key metrics from customer feedback.

        Example 1:
        Feedback: "Love the speed! Processes 1000 rows in under 2 seconds."
        Metrics: {{"speed": "2 seconds", "capacity": "1000 rows"}}

        Example 2:
        Feedback: "Interface is clean but takes 30 minutes to learn the basics."
        Metrics: {{"learning_time": "30 minutes", "ui_quality": "clean"}}

        Example 3:
        Feedback: "Costs $50/month which is reasonable for the features."
        Metrics: {{"price": "$50/month", "value": "reasonable"}}

        Now extract from:
        Feedback: "{text}"
        Metrics:"""
    
    # ═══════════════════════════════════════════════════════════
    # STRUCTURED OUTPUT PROMPTS
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def get_json_prompt(data_description: str, fields: List[str]) -> str:
        """
        Prompt that requests JSON output
        
        Args:
            data_description: What to analyze
            fields: Required JSON fields
        """
        fields_str = ", ".join([f'"{f}"' for f in fields])
        
        return f"""Analyze the following and return ONLY valid JSON with these exact fields: {fields_str}

        Data: {data_description}

        Output format:
        {{
        {chr(10).join([f'  "{field}": "..."' for field in fields])}
        }}

        Return ONLY the JSON, no other text."""
    
    @staticmethod
    def get_csv_prompt(data: str, columns: List[str]) -> str:
        """Prompt that requests CSV output"""
        return f"""Convert this data to CSV format with columns: {", ".join(columns)}

                Data: {data}

                Output format (include header):
                {",".join(columns)}
                row1_val1,row1_val2,...
                row2_val1,row2_val2,...

                Return ONLY the CSV, no other text."""
    
    # ═══════════════════════════════════════════════════════════
    # CHAIN-OF-THOUGHT PROMPTS
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def chain_of_thought_analysis(question: str) -> str:
        """Prompt that encourages step-by-step reasoning"""
        return f"""Question: {question}

        Let's approach this step-by-step:

        Step 1: What information do we have?
        Step 2: What are we trying to determine?
        Step 3: What analysis method should we use?
        Step 4: What's the conclusion?

        Please work through each step."""
    
    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def format_with_context(prompt: str, context: Dict[str, Any]) -> str:
        """
        Fill in a prompt template with context
        
        Args:
            prompt: Prompt with {placeholders}
            context: Dict of placeholder values
            
        Returns:
            Formatted prompt
        """
        return prompt.format(**context)


# Quick test
if __name__ == "__main__":
    print("🧪 Testing Prompt Templates\n")
    
    templates = PromptTemplates()
    
    # Test 1: System prompts
    print("=" * 60)
    print("DATA ANALYST SYSTEM PROMPT:")
    print("=" * 60)
    print(templates.data_analyst_system())
    print()
    
    # Test 2: Data analysis prompt
    print("=" * 60)
    print("DATA ANALYSIS PROMPT:")
    print("=" * 60)
    stats = {
        "mean": 4.2,
        "median": 4.5,
        "std": 0.8,
        "min": 2.0,
        "max": 5.0
    }
    print(templates.analyze_data_summary("Customer satisfaction ratings", stats))
    print()
    
    # Test 3: JSON output prompt
    print("=" * 60)
    print("JSON OUTPUT PROMPT:")
    print("=" * 60)
    print(templates.get_json_prompt(
        "Customer feedback: 'Great product but expensive'",
        ["sentiment", "main_topic", "price_concern"]
    ))
    print()
    
    print("✅ All templates loaded successfully!")