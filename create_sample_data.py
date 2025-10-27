
import pandas as pd
from pathlib import Path

# Ensure data directory exists
Path('data').mkdir(exist_ok=True)

# Dataset 1: E-commerce products
products = pd.DataFrame({
    'product_id': range(1, 21),
    'product_name': [
        'Laptop Pro', 'Wireless Mouse', 'Keyboard RGB', 'Monitor 27"', 'Webcam HD',
        'USB Cable', 'Headphones', 'Phone Case', 'Tablet Stand', 'Power Bank',
        'HDMI Cable', 'USB Hub', 'Laptop Bag', 'Mouse Pad', 'Phone Charger',
        'Screen Protector', 'Stylus Pen', 'Cable Organizer', 'Desk Lamp', 'Phone Holder'
    ],
    'category': [
        'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics',
        'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Electronics',
        'Accessories', 'Accessories', 'Accessories', 'Accessories', 'Accessories',
        'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Accessories'
    ],
    'price': [
        1299.99, 29.99, 89.99, 449.99, 79.99,
        9.99, 149.99, 19.99, 34.99, 49.99,
        12.99, 39.99, 59.99, 14.99, 24.99,
        9.99, 29.99, 12.99, 44.99, 19.99
    ],
    'rating': [
        4.5, 4.2, 4.7, 4.8, 4.1,
        3.9, 4.6, 4.3, 4.4, 4.5,
        4.0, 4.2, 4.6, 4.1, 4.3,
        3.8, 4.4, 4.2, 4.5, 4.0
    ],
    'reviews_count': [
         1250, 890, 670, 1250, 890,
        670, 420, 560, 340, 980,
        230, 410, 520, 780, 650,
        290, 340, 510, 180, 390
    ],
    'stock': [
        45, 120, 85, 30, 65,
        200, 50, 150, 90, 110,
        180, 75, 60, 95, 140,
        210, 100, 130, 55, 170
    ],
    'sales_last_month': [
        89, 234, 156, 67, 112,
        445, 198, 378, 201, 267,
        334, 189, 145, 223, 312,
        401, 178, 256, 134, 289
    ]
})

products.to_csv('data/products.csv', index=False)
print("✓ Created data/products.csv")

# Dataset 2: Customer reviews
reviews = pd.DataFrame({
    'review_id': range(1, 51),
    'product_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 5,
    'rating': [5, 4, 5, 4, 5, 5, 5, 4, 4, 5,
               4, 5, 4, 4, 5, 4, 5, 5, 4, 4,
               5, 4, 4, 5, 5, 4, 5, 4, 5, 4,
               4, 5, 5, 4, 4, 5, 5, 4, 4, 5,
               5, 4, 4, 5, 5, 4, 5, 4, 4, 5],
    'review_text': [
        'Amazing laptop! Super fast and great build quality.',
        'Good but a bit pricey. Performance is excellent though.',
        'Best mouse I\'ve ever used. Very comfortable.',
        'Works well, battery life could be better.',
        'RGB lighting is stunning! Great for gaming.',
        'Solid keyboard, keys feel great.',
        'Perfect monitor for work. Colors are accurate.',
        'Good monitor but stand is a bit wobbly.',
        'Clear video quality. Easy to set up.',
        'Does the job. Nothing special but works.',
    ] * 5,
    'verified_purchase': [True, True, True, False, True, True, True, True, False, True] * 5,
    'helpful_votes': [45, 23, 67, 12, 89, 34, 56, 28, 41, 19] * 5
})

reviews.to_csv('data/reviews.csv', index=False)
print("✓ Created data/reviews.csv")

# Dataset 3: Sales data
import random
from datetime import datetime, timedelta

dates = [(datetime(2024, 10, 1) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(90)]

sales = []
for date in dates:
    for _ in range(random.randint(10, 30)):
        sales.append({
            'date': date,
            'product_id': random.randint(1, 20),
            'quantity': random.randint(1, 5),
            'customer_type': random.choice(['New', 'Returning', 'VIP']),
            'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
            'discount': random.choice([0, 0.05, 0.10, 0.15, 0.20])
        })

sales_df = pd.DataFrame(sales)
sales_df.to_csv('data/sales.csv', index=False)
print(f"✓ Created data/sales.csv ({len(sales_df)} records)")

print("\n✅ All sample datasets created!")
print("\nDatasets:")
print(f"  - data/products.csv (20 products)")
print(f"  - data/reviews.csv (50 reviews)")
print(f"  - data/sales.csv ({len(sales_df)} sales records)")