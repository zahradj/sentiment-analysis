import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Load Dataset (Replace with actual dataset path)
df = pd.read_csv("reviews.csv")  # Ensure dataset contains 'Review' and 'Sentiment' columns

# Data Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = nltk.word_tokenize(text)
    text = ' '.join([word for word in words if word not in stop_words])  # Remove stopwords
    return text

df['Cleaned_Review'] = df['Review'].apply(clean_text)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Review'], df['Sentiment'], test_size=0.2, random_state=42)

# Text Vectorization and Model Training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Data Visualization
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
#Added Sentiment Analysis Script
