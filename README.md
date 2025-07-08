# Preface

Originally this was a task assigned to me for an employment opportunity. I've since decided to revamp this project into a modern approach, and focused primarily on trying to explain what this is and how it works - serving as a more educational resource for anybody interested in such topics (note that I am a beginner).

# Hotel Review Sentiment Analysis

A modern, production-ready sentiment analysis system for hotel reviews that automatically classifies customer feedback as positive or negative sentiment. This project demonstrates advanced natural language processing (NLP) and machine learning preprocessing techniques.

## 🎯 What is Sentiment Analysis?

**Sentiment Analysis** is a branch of Natural Language Processing (NLP) that determines the emotional tone or attitude expressed in text. In the context of hotel reviews, it helps businesses:

- **Understand Customer Satisfaction**: Automatically categorize thousands of reviews as positive or negative
- **Monitor Brand Reputation**: Track sentiment trends over time
- **Improve Services**: Identify common themes in negative feedback
- **Automate Review Processing**: Replace manual review categorization with AI

### Traditional Approach vs. This Approach

**Manual Processing**: 
- ❌ Time-consuming human review of each comment
- ❌ Inconsistent categorization between reviewers
- ❌ Cannot scale to thousands of reviews

**This AI Approach**:
- ✅ Process 26,000+ reviews in seconds
- ✅ Consistent, objective classification criteria
- ✅ Scalable to any dataset size
- ✅ Configurable sensitivity thresholds

## 📊 The Dataset: Hotel Booking Reviews

This dataset contains **26,386 real hotel reviews** scraped from Booking.com, providing a rich source of authentic customer feedback.

### Data Structure
```
📋 Dataset Overview:
├── 26,386 hotel reviews
├── 15 data columns
├── Reviews from multiple countries
├── Ratings from 1.0 to 10.0 scale
└── Raw text reviews + metadata
```

### Key Data Columns

| Column | Description | Example |
|--------|-------------|---------|
| `review_text` | Customer's written review | "The hotel was clean and staff friendly..." |
| `rating` | Numerical rating (1-10) | 8.5 |
| `hotel_name` | Name of the hotel | "Villa Pura Vida" |
| `nationality` | Reviewer's country | "Belgium" |
| `reviewed_at` | Date of review | "11 July 2021" |

### Sample Review Data
```
Hotel: Villa Pura Vida
Rating: 8.5/10
Review: "Everything was perfect! Quiet, cozy place to relax. 
         The breakfast was excellent and the staff was very helpful..."
Nationality: Poland
Date: July 2021
```

### Rating Distribution Analysis
```
📈 Rating Statistics:
├── Range: 1.0 - 10.0
├── Average: 8.45/10
├── Negative (< 5): 462 reviews (1.8%)
├── Neutral (5-7): 6,725 reviews (25.5%)
└── Positive (7+): 19,199 reviews (72.7%)
```

## 🔍 What I Expected vs. What I Got

### Initial Expectations
I expected to find a balanced distribution of positive, neutral, and negative reviews, similar to typical product review datasets (roughly 60% positive, 25% neutral, 15% negative).

### Actual Results
My analysis revealed a **highly positive-skewed dataset**:

```
🎯 Expected Distribution:
├── Positive: ~60%
├── Neutral: ~25%
└── Negative: ~15%

📊 Actual Distribution:
├── Positive: 95.6% (25,198 reviews)
└── Negative: 4.4% (1,165 reviews)
```

### Why This Happens
1. **Selection Bias**: People are more likely to review when they have extreme experiences
2. **Platform Effect**: Booking.com may pre-filter very negative reviews
3. **Hotel Quality**: Dataset may focus on higher-rated establishments
4. **Review Incentives**: Hotels may encourage satisfied customers to review

### Machine Learning Implications
This imbalanced dataset presents classic ML challenges:
- **Class Imbalance**: Need techniques like stratified sampling
- **Model Bias**: Risk of always predicting "positive"
- **Evaluation Metrics**: Accuracy alone is misleading (95.6% by always guessing positive)
- **Real-world Value**: Better at detecting rare negative sentiment

## 🏗️ Project Architecture

```
SentimentAnalysis/
├── 📄 DataPreprocess.py       # Main preprocessing pipeline
├── 🎯 example_usage.py        # Usage demonstration
├── 📈 analyze_ratings.py      # Data distribution analysis
├── �️ sentiment_gui.py        # Interactive GUI application
├── 🚀 launch_gui.py           # GUI launcher script
├── �📋 requirements.txt        # Python dependencies
├── 📊 booking_reviews copy.csv # Hotel reviews dataset
└── 📖 README.md               # This documentation
```

### Core Components

#### 1. **DataPreprocess.py** - The Heart of the System
Modern object-oriented preprocessing pipeline featuring:
- **Automatic column detection** for any CSV structure
- **Advanced text cleaning** (HTML, URLs, punctuation)
- **Smart sentiment classification** with configurable thresholds
- **Robust error handling** and validation
- **Professional logging** and type hints

#### 2. **sentiment_gui.py** - Interactive GUI Application 🆕
User-friendly graphical interface featuring:
- **Real-time sentiment analysis** with confidence scores
- **Detailed explanations** of why text was classified as positive/negative
- **Interactive model training** with progress indicators
- **Example reviews** to test the system
- **Model performance metrics** and statistics
- **Custom data loading** for your own CSV files

#### 3. **example_usage.py** - Quick Start Demo
Interactive demonstration showing:
- Complete preprocessing workflow
- Sample output and statistics
- Performance metrics
- Ready-to-use ML data

#### 4. **analyze_ratings.py** - Data Exploration
Comprehensive analysis tool for:
- Rating distribution visualization
- Column structure examination
- Data quality assessment
- Statistical summaries

#### 5. **demo_gui.py** - GUI Features Preview
Demonstration script showing:
- GUI capabilities overview
- Feature explanations
- Example analysis output
- Perfect for headless environments

## 🚀 Quick Start

### GUI Application (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive GUI
python launch_gui.py
```

The GUI provides:
- 🎯 **Interactive Analysis**: Type any hotel review and get instant sentiment analysis
- 🧠 **AI Explanations**: Detailed breakdown of why the AI made its decision
- 📊 **Model Training**: Train the AI model on your data with progress tracking
- 📈 **Performance Metrics**: See how well the model performs
- 💡 **Example Reviews**: Try pre-loaded examples to see how it works

### Command Line Usage
```bash
# Run the complete analysis
python example_usage.py

# Explore data distribution
python analyze_ratings.py
```

### Programming Interface
```python
from DataPreprocess import ReviewDataPreprocessor

# Initialize and process
preprocessor = ReviewDataPreprocessor('booking_reviews copy.csv')
X, y = preprocessor.prepare_data()

# Results: X = processed text, y = sentiment labels
print(f"Dataset: {len(X)} reviews")
print(f"Positive sentiment: {y.mean():.1%}")
```

## �️ Interactive GUI Features

The sentiment analysis GUI provides a comprehensive, user-friendly interface for analyzing hotel reviews with detailed explanations.

### 🎯 Key GUI Features

#### **1. Intelligent Sentiment Analysis**
- **Real-time Analysis**: Type any hotel review and get instant sentiment classification
- **Confidence Scores**: See how confident the AI is in its prediction (0-100%)
- **Visual Results**: Clear positive/negative indicators with color coding

#### **2. AI Explanation System** 
- **Word-level Analysis**: See which specific words influenced the decision
- **Impact Scores**: Understand how much each word contributed to the final sentiment
- **Model Transparency**: Detailed breakdown of the AI's decision-making process

#### **3. Interactive Model Training**
- **One-click Training**: Train the AI model on 26,000+ hotel reviews
- **Progress Tracking**: Real-time progress bar during model training
- **Performance Metrics**: See accuracy, precision, recall, and confusion matrix
- **Custom Data**: Load your own CSV files for analysis

#### **4. Example Reviews & Testing**
- **Pre-loaded Examples**: Try positive, negative, and neutral review samples
- **Custom Input**: Analyze any hotel review text you want to test
- **Processed Text View**: See how the AI cleans and processes your input

#### **5. Educational Value**
- **Step-by-step Explanations**: Learn how sentiment analysis works
- **Feature Importance**: Understand which words matter most
- **Model Architecture**: See the technical details behind the predictions

### 📱 GUI Screenshots & Workflow

```
🖥️ Main Interface Layout:
├── 🎛️ Control Panel: Train model, load data, view status
├── ✍️ Input Section: Enter reviews, load examples
├── 📊 Analysis Tab: Sentiment results and confidence
├── 🧠 Explanation Tab: Why this sentiment? (Word analysis)
└── 📈 Model Info Tab: Performance metrics and details
```

### 🎓 How the Explanation System Works

When you analyze a review, the GUI shows:

1. **Overall Sentiment**: Positive or Negative with confidence percentage
2. **Key Influencing Words**: 
   - ✅ Words that made it seem positive (e.g., "excellent", "friendly", "clean")
   - ❌ Words that made it seem negative (e.g., "terrible", "dirty", "rude")
3. **Impact Scores**: Numerical values showing how much each word mattered
4. **Processing Steps**: How the raw text was cleaned and prepared
5. **Model Details**: Technical information about the AI algorithm

### 💡 Example Analysis

**Input Review**: "The hotel was absolutely fantastic! Great location and friendly staff."

**AI Analysis**:
- 😊 **Sentiment**: POSITIVE (89.2% confidence)
- ✅ **Key Positive Words**: "fantastic" (+0.245), "great" (+0.156), "friendly" (+0.134)
- 🧠 **Explanation**: The model detected strong positive language with words like "fantastic" and "great" that are highly associated with positive hotel experiences in the training data.

## �🔧 Technical Features

### Modern Python Architecture
- **Object-Oriented Design**: Clean, maintainable class structure
- **Type Hints**: Full static type checking support
- **Error Handling**: Graceful failure with meaningful messages
- **Logging**: Structured debug information
- **Documentation**: Comprehensive docstrings

### Advanced Text Preprocessing
```python
# What the preprocessing does:
"<p>Great hotel! Visit https://example.com</p>" 
    ↓
"great hotel visit"

# Removes: HTML tags, URLs, punctuation, stopwords, short words
# Keeps: Meaningful content words for sentiment analysis
```

### Smart Data Handling
- **Column Auto-Detection**: Works with any CSV structure
- **Missing Data**: Robust handling of null/invalid entries
- **Rating Flexibility**: Configurable sentiment thresholds
- **Stratified Splitting**: Maintains class balance in train/test

## 📈 Results & Performance

### Preprocessing Output
```
✅ Successfully processed: 26,363 reviews
📊 Sentiment distribution: 95.6% positive, 4.4% negative  
🔧 Train/test split: 21,090 / 5,273 samples
⚡ Processing time: ~30 seconds
```

### Sample Processed Text
```
Original: "The hotel was absolutely fantastic! Great location near the beach. 
           Staff were super helpful. Would definitely recommend! 😊"
           
Processed: "hotel absolutely fantastic great location near beach staff 
            super helpful would definitely recommend"
```

## 🎯 Next Steps: Building ML Models

The preprocessed data is ready for machine learning:

### 1. **Text Vectorization**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X_train)
```

### 2. **Model Training**
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Handles feature interactions
- **SVM**: Good for text classification
- **Neural Networks**: LSTM/BERT for advanced performance

### 3. **Handling Class Imbalance**
- **SMOTE**: Synthetic minority oversampling
- **Class weights**: Penalize majority class
- **Threshold tuning**: Optimize decision boundary
- **Ensemble methods**: Combine multiple approaches

## � Troubleshooting

| Issue | Solution |
|-------|----------|
| NLTK download fails | Script includes fallback stopword lists |
| Column not found | Use `analyze_ratings.py` to check structure |
| Memory issues | Process data in chunks for large datasets |
| Encoding errors | Ensure CSV is UTF-8 encoded |

### GUI-Specific Issues

**Problem**: "GUI won't launch on remote server"  
**Solution**: The GUI requires a graphical display. Use the command-line tools instead: `python example_usage.py`

**Problem**: "Model training takes too long"  
**Solution**: Training on 26,000 reviews takes 30-60 seconds on modern hardware. The progress bar shows activity.

**Problem**: "Analysis seems inaccurate"  
**Solution**: Remember the model is trained on hotel reviews specifically and may not work well for other domains. myself).

## 📝 Dependencies

```txt
pandas >= 2.0.0    # Data manipulation
nltk >= 3.8.0      # Natural language processing
scikit-learn >= 1.3.0  # Machine learning tools
numpy >= 1.24.0    # Numerical computing
```