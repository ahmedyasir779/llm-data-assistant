from typing import List, Dict, Any, Optional
import pandas as pd

from .llm_client import SimpleLLM
from .prompt_templates import PromptTemplates

class DataTextGenerator:
    def __init__(self):
        """Initialize with LLM client and templates"""
        self.llm = SimpleLLM()
        self.templates = PromptTemplates()
        print("✓ DataTextGenerator initialized")
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARIZATION
    # ═══════════════════════════════════════════════════════════
    
    def summarize_dataset(self, dataset_info: Dict[str, Any]) -> str:
        """
        Generate executive summary of a dataset
        
        Args:
            dataset_info: Dictionary with dataset metadata
                - name: Dataset name
                - rows: Number of rows
                - columns: Number of columns
                - column_names: List of column names
                - missing_data: Dict of columns with missing values
                
        Returns:
            Executive summary as string
        """
        prompt = f"""Create a concise executive summary of this dataset:

        Dataset Name: {dataset_info.get('name', 'Unknown')}
        Size: {dataset_info.get('rows', 0)} rows × {dataset_info.get('columns', 0)} columns

        Columns: {', '.join(dataset_info.get('column_names', []))}

        Missing Data: {dataset_info.get('missing_data', 'None')}

        Summary should include:
        1. What this dataset likely contains (infer from column names)
        2. Data quality assessment
        3. Potential use cases
        4. Any immediate concerns

        Keep it under 150 words."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.data_analyst_system()
        )
    
    def summarize_statistics(self, column_name: str, stats: Dict[str, float]) -> str:
        """
        Generate natural language summary of column statistics
        
        Args:
            column_name: Name of the column
            stats: Dictionary of statistics (mean, median, std, min, max, etc.)
            
        Returns:
            Human-readable summary
        """
        prompt = f"""Summarize these statistics in 2-3 clear sentences:

        Column: {column_name}
        Statistics:
        - Mean: {stats.get('mean', 'N/A')}
        - Median: {stats.get('median', 'N/A')}
        - Std Dev: {stats.get('std', 'N/A')}
        - Min: {stats.get('min', 'N/A')}
        - Max: {stats.get('max', 'N/A')}
        - Count: {stats.get('count', 'N/A')}

        Focus on what these numbers tell us about the data.
        Be specific and actionable."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.concise_assistant_system()
        )
    
    # ═══════════════════════════════════════════════════════════
    # INSIGHT GENERATION
    # ═══════════════════════════════════════════════════════════
    
    def generate_insights(self, data_summary: str, top_n: int = 3) -> List[str]:
        """
        Generate key insights from data summary
        
        Args:
            data_summary: Summary of the data/analysis
            top_n: Number of insights to generate
            
        Returns:
            List of insight strings
        """
        prompt = f"""Based on this data summary, generate {top_n} key insights:

        {data_summary}

        Format: Return ONLY the insights, one per line, starting with "•"
        Example:
        - Insight 1
        - Insight 2
        - Insight 3

        Be specific, actionable, and business-focused."""
        
        response = self.llm.ask(
            prompt,
            system=self.templates.data_analyst_system()
        )
        
        # Parse response into list
        insights = [line.strip() for line in response.split('\n') if line.strip().startswith('•')]
        return insights[:top_n]
    
    def explain_correlation(self, col1: str, col2: str, correlation: float) -> str:
        """
        Explain what a correlation means in plain English
        
        Args:
            col1: First column name
            col2: Second column name
            correlation: Correlation coefficient (-1 to 1)
            
        Returns:
            Plain English explanation
        """
        prompt = f"""Explain this correlation in simple business terms:

        Correlation between "{col1}" and "{col2}": {correlation:.3f}

        Provide:
        1. What this correlation value means
        2. Is it strong, moderate, or weak?
        3. What might this tell us about the relationship?
        4. One practical implication

        Keep it under 100 words, avoid technical jargon."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.concise_assistant_system()
        )
    
    # ═══════════════════════════════════════════════════════════
    # REPORT GENERATION
    # ═══════════════════════════════════════════════════════════
    
    def generate_analysis_report(
        self,
        dataset_name: str,
        key_findings: List[str],
        statistics_summary: str,
        recommendations: Optional[List[str]] = None
    ) -> str:
        """
        Generate complete analysis report
        
        Args:
            dataset_name: Name of the dataset
            key_findings: List of key findings
            statistics_summary: Summary of statistics
            recommendations: Optional list of recommendations
            
        Returns:
            Formatted report as string
        """
        findings_text = '\n'.join([f"• {finding}" for finding in key_findings])
        
        prompt = f"""Create a professional data analysis report:

        Dataset: {dataset_name}

        Key Findings:
        {findings_text}

        Statistical Summary:
        {statistics_summary}

        {'Recommendations: ' + ', '.join(recommendations) if recommendations else ''}

        Generate a well-structured report with:
        1. Executive Summary (2-3 sentences)
        2. Detailed Findings (expand on key points)
        3. Methodology Notes (what analysis was done)
        4. Recommendations (actionable next steps)

        Keep it professional but accessible. Total length: 300-400 words."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.data_analyst_system()
        )
    
    # ═══════════════════════════════════════════════════════════
    # DATA CLEANING RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════
    
    def suggest_cleaning_steps(self, data_issues: Dict[str, Any]) -> List[str]:
        """
        Generate data cleaning recommendations
        
        Args:
            data_issues: Dictionary describing data quality issues
                - missing_values: Dict of columns with missing %
                - duplicates: Number of duplicate rows
                - outliers: Dict of columns with outlier info
                - data_types: Issues with data types
                
        Returns:
            List of cleaning step recommendations
        """
        prompt = f"""Based on these data quality issues, suggest cleaning steps:

        Issues Found:
        - Missing Values: {data_issues.get('missing_values', 'None')}
        - Duplicate Rows: {data_issues.get('duplicates', 0)}
        - Outliers: {data_issues.get('outliers', 'None detected')}
        - Data Type Issues: {data_issues.get('data_types', 'None')}

        Provide 3-5 specific, actionable cleaning steps in order of priority.
        Format: numbered list
        Example:
        1. Handle missing values in X column using Y method
        2. Remove Z duplicate rows
        etc.

        Be specific about methods (drop, fill, interpolate, etc.)"""
        
        response = self.llm.ask(
            prompt,
            system=self.templates.data_analyst_system()
        )
        
        # Parse numbered list
        steps = [line.strip() for line in response.split('\n') 
                if line.strip() and line.strip()[0].isdigit()]
        return steps
    
    # ═══════════════════════════════════════════════════════════
    # QUESTION ANSWERING
    # ═══════════════════════════════════════════════════════════
    
    def answer_about_data(self, question: str, data_context: str) -> str:
        """
        Answer questions about a dataset
        
        Args:
            question: User's question
            data_context: Relevant data information/statistics
            
        Returns:
            Answer to the question
        """
        prompt = f"""Answer this question about the dataset:

        Question: {question}

        Available Data Context:
        {data_context}

        Provide a clear, direct answer based on the available data.
        If the data doesn't contain enough information, say so and suggest what data would be needed."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.data_analyst_system()
        )
    
    # ═══════════════════════════════════════════════════════════
    # COMPARISON & ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    def compare_groups(
        self,
        group1_name: str,
        group1_stats: Dict,
        group2_name: str,
        group2_stats: Dict
    ) -> str:
        """
        Compare two groups and explain differences
        
        Args:
            group1_name: Name of first group
            group1_stats: Statistics for group 1
            group2_name: Name of second group
            group2_stats: Statistics for group 2
            
        Returns:
            Comparison analysis
        """
        prompt = f"""Compare these two groups and explain the key differences:

        Group 1: {group1_name}
        {self._format_stats(group1_stats)}

        Group 2: {group2_name}
        {self._format_stats(group2_stats)}

        Provide:
        1. Main differences (2-3 points)
        2. Statistical significance of differences
        3. Practical implications
        4. One recommendation based on the comparison

        Keep it concise and actionable."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.data_analyst_system()
        )
    
    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════
    
    def _format_stats(self, stats: Dict) -> str:
        """Format statistics dictionary for prompts"""
        return '\n'.join([f"  - {k}: {v}" for k, v in stats.items()])
    
    def generate_column_description(self, column_name: str, sample_values: List) -> str:
        """
        Generate description of what a column contains
        
        Args:
            column_name: Name of the column
            sample_values: 5-10 sample values from the column
            
        Returns:
            Description of column purpose/content
        """
        prompt = f"""Based on the column name and sample values, describe what this column contains:

            Column Name: {column_name}
            Sample Values: {', '.join([str(v) for v in sample_values[:10]])}

            Provide:
            1. What this column represents
            2. Data type and format
            3. Likely use in analysis
            4. Any quality concerns from samples

            Keep it under 80 words."""
        
        return self.llm.ask(
            prompt,
            system=self.templates.concise_assistant_system()
        )


# Quick test
if __name__ == "__main__":
    print(" Testing Text Generation\n")
    
    generator = DataTextGenerator()
    
    # Test 1: Dataset summary
    print("=" * 60)
    print("TEST 1: Dataset Summary")
    print("=" * 60)
    
    dataset_info = {
        'name': 'Customer Reviews',
        'rows': 1000,
        'columns': 5,
        'column_names': ['product_id', 'rating', 'review_text', 'date', 'verified_purchase'],
        'missing_data': {'review_text': '5%', 'verified_purchase': '2%'}
    }
    
    summary = generator.summarize_dataset(dataset_info)
    print(f"\n{summary}\n")
    
    # Test 2: Statistics summary
    print("=" * 60)
    print("TEST 2: Statistics Summary")
    print("=" * 60)
    
    stats = {
        'mean': 4.2,
        'median': 4.5,
        'std': 0.8,
        'min': 1.0,
        'max': 5.0,
        'count': 1000
    }
    
    stats_summary = generator.summarize_statistics('rating', stats)
    print(f"\n{stats_summary}\n")
    
    # Test 3: Generate insights
    print("=" * 60)
    print("TEST 3: Generate Insights")
    print("=" * 60)
    
    data_summary = """
    Customer ratings show high satisfaction with mean of 4.2/5.0.
    650 five-star reviews vs 30 one-star reviews.
    Review text shows frequent mentions of 'fast shipping' and 'quality'.
    5% missing review text suggests some customers skip detailed feedback.
    """
    
    insights = generator.generate_insights(data_summary, top_n=3)
    print("\nGenerated Insights:")
    for insight in insights:
        print(insight)
    print()
    
    # Test 4: Correlation explanation
    print("=" * 60)
    print("TEST 4: Correlation Explanation")
    print("=" * 60)
    
    explanation = generator.explain_correlation('rating', 'review_length', 0.65)
    print(f"\n{explanation}\n")
    
    # Test 5: Cleaning recommendations
    print("=" * 60)
    print("TEST 5: Cleaning Recommendations")
    print("=" * 60)
    
    data_issues = {
        'missing_values': {'review_text': '5%', 'date': '2%'},
        'duplicates': 15,
        'outliers': {'rating': '3 values outside 3 std devs'},
        'data_types': 'date column stored as string'
    }
    
    steps = generator.suggest_cleaning_steps(data_issues)
    print("\nRecommended Cleaning Steps:")
    for step in steps:
        print(step)
    print()
    
    print("=" * 60)
    print(" All tests complete!")
    print("=" * 60)