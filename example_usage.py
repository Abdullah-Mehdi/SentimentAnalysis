#!/usr/bin/env python3
"""
Example usage of the modernized sentiment analysis preprocessing module.
"""

from DataPreprocess import ReviewDataPreprocessor

def run_example():
    """Run the example preprocessing pipeline."""
    print("=== Hotel Review Sentiment Analysis - Data Preprocessing ===\n")
    
    # Initialize the preprocessor
    preprocessor = ReviewDataPreprocessor('booking_reviews copy.csv')
    
    try:
        # Load and process the data
        print("1. Loading data...")
        df = preprocessor.load_data()
        
        print("\n2. Preparing data for machine learning...")
        X, y = preprocessor.prepare_data()
        
        print("\n3. Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Display results
        print(f"\n=== Results ===")
        print(f"Dataset shape: {df.shape}")
        print(f"After preprocessing: {len(X)} samples")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Positive sentiment ratio: {y.mean():.1%}")
        
        # Show examples
        print(f"\n=== Sample Preprocessed Reviews ===")
        for i, (text, label) in enumerate(zip(X_train.head(3), y_train.head(3))):
            sentiment = "😊 Positive" if label else "😞 Negative"
            print(f"\n{i+1}. [{sentiment}]")
            print(f"   Text: {text[:150]}{'...' if len(text) > 150 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = run_example()
    if success:
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"💡 You can now use X_train, X_test, y_train, y_test for machine learning models.")
    else:
        print(f"\n❌ Preprocessing failed. Please check the error messages above.")
