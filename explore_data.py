#!/usr/bin/env python3
"""
Data exploration script to understand the hotel reviews dataset.
"""

import pandas as pd

def explore_data():
    """Explore the hotel reviews dataset."""
    df = pd.read_csv('booking_reviews copy.csv')
    
    print('=== HOTEL REVIEWS DATASET OVERVIEW ===\n')
    print(f'📊 Dataset Size: {len(df):,} reviews')
    print(f'📋 Columns: {len(df.columns)} features\n')
    
    print('=== KEY COLUMNS ===')
    key_columns = ['review_text', 'rating', 'hotel_name', 'nationality', 'reviewed_at']
    for col in key_columns:
        if col in df.columns:
            print(f'✓ {col}')
    print()
    
    print('=== SAMPLE REVIEW ===')
    sample_review = df['review_text'].iloc[0]
    sample_rating = df['rating'].iloc[0]
    sample_hotel = df['hotel_name'].iloc[0]
    
    print(f'Hotel: {sample_hotel}')
    print(f'Rating: {sample_rating}/10')
    print(f'Review: "{sample_review[:150]}..."')
    print()
    
    print('=== RATING DISTRIBUTION ===')
    print(f'Rating Range: {df["rating"].min()} - {df["rating"].max()}')
    print(f'Average Rating: {df["rating"].mean():.2f}')
    
    negative = (df['rating'] < 5).sum()
    neutral = ((df['rating'] >= 5) & (df['rating'] < 7)).sum()
    positive = (df['rating'] >= 7).sum()
    
    print(f'Negative (< 5): {negative:,} ({negative/len(df)*100:.1f}%)')
    print(f'Neutral (5-7): {neutral:,} ({neutral/len(df)*100:.1f}%)')
    print(f'Positive (7+): {positive:,} ({positive/len(df)*100:.1f}%)')

if __name__ == "__main__":
    explore_data()
