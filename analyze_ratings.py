#!/usr/bin/env python3
"""
Analyze the rating distribution in the dataset to better understand the data.
"""

import pandas as pd
import numpy as np

def analyze_ratings():
    """Analyze the rating distribution in the dataset."""
    print("=== Rating Distribution Analysis ===\n")
    
    # Load the data
    df = pd.read_csv('booking_reviews copy.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Check rating columns
    rating_cols = [col for col in df.columns if 'rating' in col.lower()]
    print(f"Rating columns found: {rating_cols}\n")
    
    for col in rating_cols:
        print(f"=== {col} ===")
        values = pd.to_numeric(df[col], errors='coerce')
        print(f"Valid values: {values.notna().sum()}")
        print(f"Range: {values.min()} - {values.max()}")
        print(f"Mean: {values.mean():.2f}")
        print(f"Distribution:")
        print(values.value_counts().sort_index())
        print()

if __name__ == "__main__":
    analyze_ratings()
