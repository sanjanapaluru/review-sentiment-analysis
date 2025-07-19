"""
Generate sample data for customer review sentiment analysis.
"""

import pandas as pd
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Sample reviews with sentiment labels
sample_reviews = [
    ("This product is amazing! I love it so much.", "positive"),
    ("Terrible quality, waste of money.", "negative"),
    ("Great value for money, highly recommend.", "positive"),
    ("Poor customer service, disappointed.", "negative"),
    ("Excellent product, exceeded my expectations.", "positive"),
    ("Not worth the price, very disappointed.", "negative"),
    ("Perfect for my needs, very satisfied.", "positive"),
    ("Broke after one day, terrible quality.", "negative"),
    ("Outstanding quality and fast delivery.", "positive"),
    ("Completely useless, don't buy this.", "negative"),
    ("Amazing customer service and great product.", "positive"),
    ("Worst purchase ever, total waste of money.", "negative"),
    ("Good product but could be better.", "positive"),
    ("Disappointed with the quality.", "negative"),
    ("Love this product, using it daily.", "positive"),
    ("Returned it immediately, poor quality.", "negative"),
    ("Highly recommend to everyone.", "positive"),
    ("Not what I expected, very disappointed.", "negative"),
    ("Best purchase I've made in years.", "positive"),
    ("Cheap quality, not worth buying.", "negative"),
    ("Fantastic product, great value.", "positive"),
    ("Terrible experience, avoid this product.", "negative"),
    ("Works perfectly, exactly as described.", "positive"),
    ("Broke within a week, poor build quality.", "negative"),
    ("Excellent service and fast shipping.", "positive"),
    ("Would not recommend to anyone.", "negative"),
    ("Great product, very happy with purchase.", "positive"),
    ("Complete waste of time and money.", "negative"),
    ("Perfect quality, exactly what I needed.", "positive"),
    ("Disappointed with the overall experience.", "negative"),
    ("Amazing quality and great customer support.", "positive"),
    ("Poor packaging, product arrived damaged.", "negative"),
    ("Excellent value for the price.", "positive"),
    ("Not satisfied with the product quality.", "negative"),
    ("Outstanding product, highly recommend.", "positive"),
    ("Very poor quality, returned immediately.", "negative"),
    ("Great product, using it every day.", "positive"),
    ("Terrible customer service experience.", "negative"),
    ("Perfect fit, exactly as advertised.", "positive"),
    ("Disappointed with the purchase.", "negative"),
    ("Fantastic quality and great price.", "positive"),
    ("Would not buy again, poor quality.", "negative"),
    ("Excellent product, very satisfied.", "positive"),
    ("Not worth the money, disappointed.", "negative"),
    ("Amazing product, exceeded expectations.", "positive"),
    ("Poor quality, not as described.", "negative"),
    ("Great value, highly recommend.", "positive"),
    ("Terrible experience, avoid this.", "negative"),
    ("Perfect product, love it.", "positive"),
    ("Completely disappointed with quality.", "negative")
]

# Create DataFrame
df = pd.DataFrame(sample_reviews, columns=['review_text', 'sentiment'])

# Save to CSV
df.to_csv('data/sample_reviews.csv', index=False)

print(f"Sample data created: {len(df)} reviews")
print(f"Sentiment distribution:")
print(df['sentiment'].value_counts())
print(f"Data saved to: data/sample_reviews.csv")
