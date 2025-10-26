import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_generator import DataTextGenerator
import json

def demo_executive_summary():
    """Generate executive summary from dataset"""
    print("\n" + "=" * 70)
    print("DEMO 1: Executive Summary Generation")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    dataset_info = {
        'name': 'E-commerce Sales Q4 2024',
        'rows': 15420,
        'columns': 12,
        'column_names': [
            'order_id', 'customer_id', 'product_category', 'price',
            'quantity', 'discount', 'shipping_cost', 'order_date',
            'delivery_date', 'customer_rating', 'payment_method', 'region'
        ],
        'missing_data': {
            'customer_rating': '8%',
            'delivery_date': '3%'
        }
    }
    
    print("\n Dataset Info:")
    print(json.dumps(dataset_info, indent=2))
    
    print("\n Generating Executive Summary...")
    summary = generator.summarize_dataset(dataset_info)
    
    print("\n" + "-" * 70)
    print("EXECUTIVE SUMMARY:")
    print("-" * 70)
    print(summary)


def demo_insights_from_metrics():
    """Generate business insights from key metrics"""
    print("\n" + "=" * 70)
    print("DEMO 2: Business Insights from Metrics")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    metrics_summary = """
    Q4 2024 Performance Metrics:
    - Total Orders: 15,420 (↑ 23% vs Q3)
    - Average Order Value: $127.50 (↑ 8% vs Q3)
    - Customer Satisfaction: 4.2/5.0 (↓ 0.1 vs Q3)
    - Return Rate: 6.5% (↑ 1.2% vs Q3)
    - Top Category: Electronics (42% of revenue)
    - Fastest Growing: Home & Garden (↑ 67% vs Q3)
    - Peak Sales Day: Black Friday ($2.1M in 24 hours)
    """
    
    print("\n Metrics:")
    print(metrics_summary)
    
    print("\n Generating Insights...")
    insights = generator.generate_insights(metrics_summary, top_n=5)
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight}")


def demo_data_quality_report():
    """Generate data quality assessment and recommendations"""
    print("\n" + "=" * 70)
    print("DEMO 3: Data Quality Assessment")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    data_issues = {
        'missing_values': {
            'customer_email': '12%',
            'phone_number': '35%',
            'delivery_date': '3%',
            'customer_rating': '8%'
        },
        'duplicates': 247,
        'outliers': {
            'price': '45 orders with price > $5000 (possible bulk orders)',
            'shipping_cost': '23 orders with shipping > $200'
        },
        'data_types': 'order_date stored as string, needs conversion to datetime'
    }
    
    print("\n Data Quality Issues Found:")
    print(json.dumps(data_issues, indent=2))
    
    print("\n Generating Cleaning Recommendations...")
    steps = generator.suggest_cleaning_steps(data_issues)
    
    print("\n" + "-" * 70)
    print("RECOMMENDED CLEANING STEPS:")
    print("-" * 70)
    for step in steps:
        print(f"  {step}")


def demo_statistical_storytelling():
    """Turn dry statistics into engaging narrative"""
    print("\n" + "=" * 70)
    print("DEMO 4: Statistical Storytelling")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    stats = {
        'mean': 127.50,
        'median': 98.00,
        'std': 85.30,
        'min': 12.99,
        'max': 4999.99,
        'count': 15420,
        'q1': 65.00,
        'q3': 145.00
    }
    
    print("\n Raw Statistics (Order Value):")
    print(json.dumps(stats, indent=2))
    
    print("\n Generating Story...")
    story = generator.summarize_statistics('order_value', stats)
    
    print("\n" + "-" * 70)
    print("STATISTICAL NARRATIVE:")
    print("-" * 70)
    print(story)


def demo_correlation_explanation():
    """Explain correlations in business terms"""
    print("\n" + "=" * 70)
    print("DEMO 5: Correlation Explanations")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    correlations = [
        ('discount_percentage', 'order_value', -0.42),
        ('customer_rating', 'delivery_speed', 0.68),
        ('review_length', 'helpful_votes', 0.55),
        ('price', 'return_rate', 0.31)
    ]
    
    for col1, col2, corr in correlations:
        print(f"\n Correlation: {col1} ↔ {col2} = {corr:.2f}")
        explanation = generator.explain_correlation(col1, col2, corr)
        print(f" Explanation:\n{explanation}")
        print("-" * 70)


def demo_comparison_analysis():
    """Compare two customer segments"""
    print("\n" + "=" * 70)
    print("DEMO 6: Segment Comparison")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    premium_stats = {
        'avg_order_value': 245.30,
        'order_frequency': 4.2,
        'satisfaction': 4.5,
        'return_rate': '4.2%',
        'lifetime_value': 3850
    }
    
    regular_stats = {
        'avg_order_value': 95.60,
        'order_frequency': 2.1,
        'satisfaction': 4.1,
        'return_rate': '7.8%',
        'lifetime_value': 980
    }
    
    print("\n Premium Customers:")
    print(json.dumps(premium_stats, indent=2))
    
    print("\n Regular Customers:")
    print(json.dumps(regular_stats, indent=2))
    
    print("\n Generating Comparison Analysis...")
    analysis = generator.compare_groups(
        'Premium Customers',
        premium_stats,
        'Regular Customers',
        regular_stats
    )
    
    print("\n" + "-" * 70)
    print("COMPARISON ANALYSIS:")
    print("-" * 70)
    print(analysis)


def demo_full_report():
    """Generate complete analysis report"""
    print("\n" + "=" * 70)
    print("DEMO 7: Complete Analysis Report")
    print("=" * 70)
    
    generator = DataTextGenerator()
    
    key_findings = [
        "Revenue increased 23% QoQ driven by Black Friday surge",
        "Customer satisfaction slightly decreased despite higher sales",
        "Return rate increased, possibly due to rushed holiday deliveries",
        "Electronics dominates but Home & Garden shows strongest growth",
        "Premium customers have 4x higher lifetime value"
    ]
    
    stats_summary = """
    15,420 orders analyzed across Q4 2024. Average order value $127.50,
    with median at $98. 8% missing customer ratings. 247 duplicate records found.
    Strong correlation (0.68) between delivery speed and satisfaction.
    """
    
    recommendations = [
        "Improve delivery logistics to boost satisfaction",
        "Invest in Home & Garden category expansion",
        "Implement premium customer loyalty program",
        "Address data quality issues in customer contact info"
    ]
    
    print("\n Generating Full Report...")
    report = generator.generate_analysis_report(
        'E-commerce Q4 2024 Analysis',
        key_findings,
        stats_summary,
        recommendations
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE ANALYSIS REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TEXT GENERATION DEMONSTRATIONS")
    print("Real-world data analysis use cases")
    print("=" * 70)
    
    # Run all demos
    demo_executive_summary()
    demo_insights_from_metrics()
    demo_data_quality_report()
    demo_statistical_storytelling()
    demo_correlation_explanation()
    demo_comparison_analysis()
    demo_full_report()
    
    print("\n" + "=" * 70)
    print("✅ ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("  1. LLMs can turn data into narratives")
    print("  2. Automate report generation")
    print("  3. Make insights accessible to non-technical stakeholders")
    print("  4. Scale analysis documentation")
    print("  5. Consistent quality across reports")