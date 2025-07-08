"""
Modern (2025) Sentiment Analysis Data Preprocessing Module

This module provides functionality for preprocessing hotel review data
for sentiment analysis using modern Python practices and libraries.
"""

import pandas as pd
import nltk
import re
import string
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewDataPreprocessor:
    """A class for preprocessing hotel review data for sentiment analysis."""
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the preprocessor with a CSV file path.
        
        Args:
            csv_file_path (str): Path to the CSV file containing review data
        """
        self.csv_file_path = Path(csv_file_path)
        self.df: Optional[pd.DataFrame] = None
        self.stop_words = None
        self._download_nltk_data()
        
    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
            logger.info("NLTK data downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
            # Fallback to basic stopwords if NLTK fails
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before',
                'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off',
                'over', 'under', 'again', 'further', 'then', 'once'
            }
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        try:
            if not self.csv_file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
            
            self.df = pd.read_csv(self.csv_file_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags (for web-scraped reviews)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)
    
    def detect_text_column(self) -> str:
        """
        Automatically detect the review text column.
        
        Returns:
            str: Name of the detected text column
            
        Raises:
            ValueError: If no suitable text column is found
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Common patterns for review text columns
        text_column_patterns = [
            'review_text', 'review', 'text', 'comment', 'feedback',
            'Review Text Hotel Location', 'review_content', 'description'
        ]
        
        for pattern in text_column_patterns:
            for col in self.df.columns:
                if pattern.lower() in col.lower():
                    logger.info(f"Detected text column: {col}")
                    return col
        
        # If no pattern matches, look for string columns with long text
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                avg_length = self.df[col].astype(str).str.len().mean()
                if avg_length > 50:  # Assuming reviews are longer than 50 characters
                    logger.info(f"Detected text column based on length: {col}")
                    return col
        
        raise ValueError("Could not detect a suitable text column")
    
    def detect_rating_column(self) -> str:
        """
        Automatically detect the rating column.
        
        Returns:
            str: Name of the detected rating column
            
        Raises:
            ValueError: If no suitable rating column is found
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Common patterns for rating columns (prioritized order)
        rating_column_patterns = [
            'rating', 'score', 'stars', 'Review Rating', 'rate', 'avg_rating'
        ]
        
        for pattern in rating_column_patterns:
            for col in self.df.columns:
                if pattern.lower() in col.lower():
                    # Check if it's numeric and has reasonable rating range
                    try:
                        values = pd.to_numeric(self.df[col], errors='coerce')
                        if not values.isna().all():
                            min_val, max_val = values.min(), values.max()
                            if 0 <= min_val and max_val <= 10:  # Reasonable rating range
                                logger.info(f"Detected rating column: {col}")
                                return col
                    except:
                        continue
        
        raise ValueError("Could not detect a suitable rating column")
    
    def prepare_data(self, text_column: Optional[str] = None, 
                    rating_column: Optional[str] = None,
                    rating_threshold: float = 7.0) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for machine learning.
        
        Args:
            text_column (str, optional): Name of the text column. Auto-detected if None.
            rating_column (str, optional): Name of the rating column. Auto-detected if None.
            rating_threshold (float): Threshold for binary classification (default: 7.0)
            
        Returns:
            Tuple[pd.Series, pd.Series]: Preprocessed features (X) and labels (y)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Auto-detect columns if not provided
        if text_column is None:
            text_column = self.detect_text_column()
        
        if rating_column is None:
            rating_column = self.detect_rating_column()
        
        # Check if columns exist
        if text_column not in self.df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        
        if rating_column not in self.df.columns:
            raise ValueError(f"Rating column '{rating_column}' not found in dataset")
        
        # Clean the data - remove rows with missing values
        clean_df = self.df[[text_column, rating_column]].dropna()
        logger.info(f"Cleaned data shape: {clean_df.shape}")
        
        # Preprocess text
        logger.info("Preprocessing text data...")
        X = clean_df[text_column].apply(self.preprocess_text)
        
        # Convert ratings to binary labels (positive/negative sentiment)
        y = pd.to_numeric(clean_df[rating_column], errors='coerce')
        y = (y > rating_threshold).astype(int)
        
        # Remove empty texts after preprocessing
        non_empty_mask = X.str.len() > 0
        X = X[non_empty_mask]
        y = y[non_empty_mask]
        
        logger.info(f"Final dataset shape: {len(X)} samples")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X: pd.Series, y: pd.Series, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.Series): Features
            y (pd.Series): Labels
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Testing set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test


def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = ReviewDataPreprocessor('booking_reviews copy.csv')
    
    try:
        # Load data
        df = preprocessor.load_data()
        
        # Prepare data for machine learning
        X, y = preprocessor.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Example of how to use the preprocessed data
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(X)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Positive sentiment ratio: {y.mean():.2%}")
        
        # Show sample preprocessed text
        print(f"\nSample preprocessed reviews:")
        for i, (text, label) in enumerate(zip(X_train.head(3), y_train.head(3))):
            sentiment = "Positive" if label else "Negative"
            print(f"{i+1}. [{sentiment}] {text[:100]}...")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # Run the preprocessing pipeline
    X_train, X_test, y_train, y_test = main()
