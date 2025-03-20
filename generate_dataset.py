import pandas as pd

# Creating a sample dataset
data = {
    "Review": [
        "I love this product! It's amazing and works perfectly.",
        "The quality is terrible, very disappointed.",
        "It's okay, not the best but not the worst either.",
        "Absolutely fantastic! Would buy again.",
        "Waste of money. Do not recommend at all.",
        "Pretty decent, could be better.",
        "I'm very happy with this purchase, highly recommended!",
        "Not great, but not horrible either.",
        "Excellent quality, exceeded my expectations!",
        "Poor design and bad customer service."
    ],
    "Sentiment": [
        "Positive", "Negative", "Neutral", "Positive", "Negative", 
        "Neutral", "Positive", "Neutral", "Positive", "Negative"
    ]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Saving to CSV
df.to_csv("reviews.csv", index=False)

print("Sample dataset 'reviews.csv' created successfully!")