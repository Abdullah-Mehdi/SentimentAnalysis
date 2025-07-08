#!/usr/bin/env python3
"""
Hotel Review Sentiment Analysis GUI

A user-friendly graphical interface for analyzing hotel reviews with 
explanations of why the sentiment was classified as positive or negative.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from DataPreprocess import ReviewDataPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import re
from typing import List, Tuple, Dict
import threading
import queue


class SentimentAnalyzerGUI:
    """GUI application for sentiment analysis with explanations."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Hotel Review Sentiment Analyzer")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize data and model variables
        self.preprocessor = None
        self.vectorizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        
        # Queue for threading
        self.queue = queue.Queue()
        
        self.setup_gui()
        self.load_default_data()
        
    def setup_gui(self):
        """Setup the GUI layout."""
        # Create main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏨 Hotel Review Sentiment Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Input section
        self.setup_input_section(main_frame)
        
        # Results section
        self.setup_results_section(main_frame)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_control_panel(self, parent):
        """Setup the control panel with buttons."""
        control_frame = ttk.LabelFrame(parent, text="Model Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Train model button
        self.train_button = ttk.Button(control_frame, text="Train Model", 
                                      command=self.train_model_threaded)
        self.train_button.grid(row=0, column=0, padx=(0, 10))
        
        # Load data button
        load_button = ttk.Button(control_frame, text="Load CSV Data", 
                                command=self.load_custom_data)
        load_button.grid(row=0, column=1, padx=(0, 10))
        
        # Model status
        self.model_status_label = ttk.Label(control_frame, text="Status: Model not trained", 
                                           foreground="red")
        self.model_status_label.grid(row=0, column=2, padx=(20, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def setup_input_section(self, parent):
        """Setup the input section for review text."""
        input_frame = ttk.LabelFrame(parent, text="Review Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Input text area
        ttk.Label(input_frame, text="Enter a hotel review to analyze:").grid(row=0, column=0, sticky=tk.W)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=4, width=60, 
                                                   wrap=tk.WORD, font=('Arial', 10))
        self.input_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 10))
        
        # Analyze button
        self.analyze_button = ttk.Button(input_frame, text="Analyze Sentiment", 
                                        command=self.analyze_review, state="disabled")
        self.analyze_button.grid(row=2, column=0, pady=(0, 5))
        
        # Example button
        example_button = ttk.Button(input_frame, text="Load Example", 
                                   command=self.load_example)
        example_button.grid(row=2, column=1, padx=(10, 0), pady=(0, 5))
        
    def setup_results_section(self, parent):
        """Setup the results display section."""
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(results_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Analysis tab
        self.setup_analysis_tab(notebook)
        
        # Explanation tab
        self.setup_explanation_tab(notebook)
        
        # Model info tab
        self.setup_model_info_tab(notebook)
        
    def setup_analysis_tab(self, parent):
        """Setup the analysis results tab."""
        analysis_frame = ttk.Frame(parent, padding="10")
        parent.add(analysis_frame, text="Sentiment Analysis")
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(2, weight=1)
        
        # Sentiment result
        self.sentiment_frame = ttk.Frame(analysis_frame)
        self.sentiment_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.sentiment_label = ttk.Label(self.sentiment_frame, text="Sentiment: Not analyzed", 
                                        font=('Arial', 14, 'bold'))
        self.sentiment_label.grid(row=0, column=0, sticky=tk.W)
        
        self.confidence_label = ttk.Label(self.sentiment_frame, text="Confidence: -", 
                                         font=('Arial', 10))
        self.confidence_label.grid(row=1, column=0, sticky=tk.W)
        
        # Processed text
        ttk.Label(analysis_frame, text="Processed Text:").grid(row=1, column=0, sticky=tk.W)
        
        self.processed_text = scrolledtext.ScrolledText(analysis_frame, height=6, width=60, 
                                                       wrap=tk.WORD, font=('Arial', 9),
                                                       state=tk.DISABLED)
        self.processed_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        
    def setup_explanation_tab(self, parent):
        """Setup the explanation tab."""
        explanation_frame = ttk.Frame(parent, padding="10")
        parent.add(explanation_frame, text="Why This Sentiment?")
        explanation_frame.columnconfigure(0, weight=1)
        explanation_frame.rowconfigure(0, weight=1)
        
        self.explanation_text = scrolledtext.ScrolledText(explanation_frame, height=15, width=60, 
                                                         wrap=tk.WORD, font=('Arial', 10),
                                                         state=tk.DISABLED)
        self.explanation_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def setup_model_info_tab(self, parent):
        """Setup the model information tab."""
        model_frame = ttk.Frame(parent, padding="10")
        parent.add(model_frame, text="Model Information")
        model_frame.columnconfigure(0, weight=1)
        model_frame.rowconfigure(0, weight=1)
        
        self.model_info_text = scrolledtext.ScrolledText(model_frame, height=15, width=60, 
                                                        wrap=tk.WORD, font=('Arial', 9),
                                                        state=tk.DISABLED)
        self.model_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load data and train model to begin analysis")
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def load_default_data(self):
        """Load the default dataset."""
        try:
            self.preprocessor = ReviewDataPreprocessor('booking_reviews copy.csv')
            self.status_var.set("Default dataset loaded - Click 'Train Model' to begin")
        except Exception as e:
            self.status_var.set(f"Error loading default data: {str(e)}")
            
    def load_custom_data(self):
        """Load custom CSV data."""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.preprocessor = ReviewDataPreprocessor(file_path)
                self.is_trained = False
                self.model_status_label.config(text="Status: Model not trained", foreground="red")
                self.analyze_button.config(state="disabled")
                self.status_var.set(f"Loaded: {file_path} - Click 'Train Model' to begin")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
                
    def train_model_threaded(self):
        """Train the model in a separate thread."""
        if not self.preprocessor:
            messagebox.showerror("Error", "Please load data first")
            return
            
        # Start progress bar
        self.progress.start()
        self.train_button.config(state="disabled")
        self.status_var.set("Training model...")
        
        # Start training in background thread
        thread = threading.Thread(target=self.train_model_background)
        thread.daemon = True
        thread.start()
        
        # Check for completion
        self.root.after(100, self.check_training_complete)
        
    def train_model_background(self):
        """Train the model in background thread."""
        try:
            # Load and prepare data
            df = self.preprocessor.load_data()
            X, y = self.preprocessor.prepare_data()
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.split_data(X, y)
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_vec = self.vectorizer.fit_transform(self.X_train)
            
            # Train model
            self.model = LogisticRegression(random_state=42, class_weight='balanced')
            self.model.fit(X_train_vec, self.y_train)
            
            # Test model
            X_test_vec = self.vectorizer.transform(self.X_test)
            y_pred = self.model.predict(X_test_vec)
            
            # Calculate metrics
            self.model_metrics = {
                'accuracy': (y_pred == self.y_test).mean(),
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            # Signal completion
            self.queue.put("training_complete")
            
        except Exception as e:
            self.queue.put(f"training_error: {str(e)}")
            
    def check_training_complete(self):
        """Check if training is complete."""
        try:
            message = self.queue.get_nowait()
            if message == "training_complete":
                self.training_complete()
            elif message.startswith("training_error"):
                error = message.split(": ", 1)[1]
                self.training_error(error)
        except queue.Empty:
            # Continue checking
            self.root.after(100, self.check_training_complete)
            
    def training_complete(self):
        """Handle training completion."""
        self.progress.stop()
        self.train_button.config(state="normal")
        self.analyze_button.config(state="normal")
        self.is_trained = True
        
        self.model_status_label.config(text="Status: Model trained ✓", foreground="green")
        self.status_var.set("Model trained successfully - Ready for sentiment analysis")
        
        # Update model info
        self.update_model_info()
        
    def training_error(self, error):
        """Handle training error."""
        self.progress.stop()
        self.train_button.config(state="normal")
        self.status_var.set("Training failed")
        messagebox.showerror("Training Error", f"Failed to train model: {error}")
        
    def update_model_info(self):
        """Update the model information tab."""
        if not self.is_trained:
            return
            
        info_text = f"""📊 MODEL PERFORMANCE SUMMARY

Training Data: {len(self.X_train):,} reviews
Testing Data: {len(self.X_test):,} reviews
Features: {self.vectorizer.max_features:,} TF-IDF features

Accuracy: {self.model_metrics['accuracy']:.1%}

📈 DETAILED CLASSIFICATION REPORT:
{self.model_metrics['classification_report']}

🎯 CONFUSION MATRIX:
{self.model_metrics['confusion_matrix']}

🔧 MODEL DETAILS:
- Algorithm: Logistic Regression with balanced class weights
- Vectorization: TF-IDF with {self.vectorizer.max_features:,} features
- Preprocessing: HTML removal, URL cleaning, stopword filtering
- Class Balance: Stratified train/test split
"""
        
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(1.0, info_text)
        self.model_info_text.config(state=tk.DISABLED)
        
    def load_example(self):
        """Load an example review."""
        examples = [
            "The hotel was absolutely fantastic! The staff were incredibly friendly and helpful. The room was spotless and the breakfast was delicious. Perfect location near the beach. Would definitely stay here again!",
            "Terrible experience. The room was dirty, the staff was rude, and the food was awful. The WiFi didn't work and there was construction noise all night. Would never recommend this place.",
            "Average hotel. The room was okay but nothing special. Staff was friendly enough. Good location but a bit overpriced for what you get."
        ]
        
        import random
        example = random.choice(examples)
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(1.0, example)
        
    def analyze_review(self):
        """Analyze the input review."""
        if not self.is_trained:
            messagebox.showerror("Error", "Please train the model first")
            return
            
        review_text = self.input_text.get(1.0, tk.END).strip()
        if not review_text:
            messagebox.showwarning("Warning", "Please enter a review to analyze")
            return
            
        try:
            # Preprocess the text
            processed_text = self.preprocessor.preprocess_text(review_text)
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.model.predict(text_vector)[0]
            probability = self.model.predict_proba(text_vector)[0]
            
            # Update results
            self.update_analysis_results(review_text, processed_text, prediction, probability)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze review: {str(e)}")
            
    def update_analysis_results(self, original_text: str, processed_text: str, 
                               prediction: int, probability: np.ndarray):
        """Update the analysis results display."""
        # Sentiment and confidence
        sentiment = "😊 POSITIVE" if prediction == 1 else "😞 NEGATIVE"
        confidence = max(probability) * 100
        
        sentiment_color = "green" if prediction == 1 else "red"
        self.sentiment_label.config(text=f"Sentiment: {sentiment}", foreground=sentiment_color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Processed text
        self.processed_text.config(state=tk.NORMAL)
        self.processed_text.delete(1.0, tk.END)
        self.processed_text.insert(1.0, processed_text)
        self.processed_text.config(state=tk.DISABLED)
        
        # Generate explanation
        explanation = self.generate_explanation(original_text, processed_text, prediction, probability)
        
        self.explanation_text.config(state=tk.NORMAL)
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(1.0, explanation)
        self.explanation_text.config(state=tk.DISABLED)
        
        self.status_var.set(f"Analysis complete: {sentiment} ({confidence:.1f}% confidence)")
        
    def generate_explanation(self, original_text: str, processed_text: str, 
                           prediction: int, probability: np.ndarray) -> str:
        """Generate explanation for the sentiment prediction."""
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = max(probability) * 100
        
        # Get feature weights
        feature_names = self.vectorizer.get_feature_names_out()
        text_vector = self.vectorizer.transform([processed_text])
        
        # Get the most influential words
        feature_weights = self.model.coef_[0]
        word_scores = []
        
        for word_idx in text_vector.nonzero()[1]:
            word = feature_names[word_idx]
            weight = feature_weights[word_idx]
            tf_idf_score = text_vector[0, word_idx]
            influence = weight * tf_idf_score
            word_scores.append((word, influence, weight))
        
        # Sort by influence
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation
        explanation = f"""🤖 SENTIMENT ANALYSIS EXPLANATION

📋 PREDICTION: {sentiment} (Confidence: {confidence:.1f}%)

🔍 WHY THIS SENTIMENT?

The model analyzed your review and classified it as {sentiment.lower()} based on the following factors:

📊 KEY INFLUENCING WORDS:
"""
        
        # Add top influential words
        top_words = word_scores[:10]  # Top 10 most influential
        positive_words = [w for w in top_words if w[1] > 0]
        negative_words = [w for w in top_words if w[1] < 0]
        
        if positive_words:
            explanation += "\n✅ POSITIVE INDICATORS:\n"
            for word, influence, weight in positive_words[:5]:
                explanation += f"   • '{word}' (impact: +{abs(influence):.3f})\n"
        
        if negative_words:
            explanation += "\n❌ NEGATIVE INDICATORS:\n"
            for word, influence, weight in negative_words[:5]:
                explanation += f"   • '{word}' (impact: -{abs(influence):.3f})\n"
        
        explanation += f"""
🧠 HOW THE MODEL WORKS:

1. TEXT PREPROCESSING:
   • Converted to lowercase
   • Removed punctuation and HTML tags
   • Filtered out stopwords (the, and, is, etc.)
   • Kept meaningful words: {len(processed_text.split())} words

2. FEATURE EXTRACTION:
   • Used TF-IDF (Term Frequency-Inverse Document Frequency)
   • Created {self.vectorizer.max_features:,} numerical features
   • Each word gets a score based on importance

3. MACHINE LEARNING PREDICTION:
   • Logistic Regression model trained on {len(self.X_train):,} hotel reviews
   • Model learned patterns from positive/negative examples
   • Balanced for class imbalance (95.6% positive reviews in training)

📈 CONFIDENCE INTERPRETATION:
{confidence:.1f}% confidence means the model is {"very" if confidence > 80 else "moderately" if confidence > 60 else "somewhat"} certain about this prediction.

💡 REMEMBER: This is an AI prediction based on patterns in hotel review data. Context and sarcasm may not always be perfectly captured."""
        
        return explanation


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = SentimentAnalyzerGUI(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
