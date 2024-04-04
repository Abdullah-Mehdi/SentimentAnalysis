import pandas as pd # For data manipulation and analysis
from nltk.corpus import stopwords # For removing stopwords
import re # For regular expressions
import string # For string operations

# Load dataset from CSV file to Pandas DataFrame
df = pd.read_csv('booking_reviews copy.csv')

# This line stop = set(stopwords.words('english')) is used to download the stopwords corpus
stop = set(stopwords.words('english'))
# Define a function to preprocess text
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regular expressions
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove stopwords from text
    text = ' '.join([word for word in text.split() if word not in stop])
    # Return preprocessed text
    return text

# Apply the preprocess function to the 'Review Text Hotel Location' column
df['Review Text Hotel Location'] = df['Review Text Hotel Location'].apply(preprocess)

from sklearn.model_selection import train_test_split # For splitting dataset into training and testing sets
# The input features are the 'Review Text Hotel Location' column
X = df['Review Text Hotel Location']
# The output labels are the 'Review Rating' column
y = df['Review Rating'].apply(lambda x: 1 if x > 3 else 0) # Binarize ratings
# Split the dataset into training and testing sets. 20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage
# new_review = "The hotel was clean and the staff was friendly."
# sentiment = predict_sentiment(new_review)
# print(f"Sentiment of the review: {sentiment}")
