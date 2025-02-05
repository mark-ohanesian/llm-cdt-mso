import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample training data with explicit category labels
training_data = [
    {"text": "Great website, very helpful!", "category": "kudos"},
    {"text": "Excellent service, thanks!", "category": "kudos"},
    {"text": "The website is ok, it's running very slow.", "category": "problem"},
    {"text": "I can't find the information I need.", "category": "problem"},
    {"text": "This page is broken, please fix it.", "category": "problem"},
    {"text": "How do I apply for a permit?", "category": "question"},
    {"text": "Who should I contact for support?", "category": "question"},
]

# Separate text samples and categories
feedback_samples = [item["text"] for item in training_data]
categories = [item["category"] for item in training_data]

# Create a pipeline: text vectorization + Naïve Bayes model
model_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model_pipeline.fit(feedback_samples, categories)

# Save the model locally
with open("feedback_classifier.pkl", "wb") as model_file:
    pickle.dump(model_pipeline, model_file)

print("Model trained and saved successfully!")

# Function to classify new feedback
def classify_feedback(user_input):
    with open("feedback_classifier.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    category = model.predict([user_input])[0]
    return category

# Example usage
if __name__ == "__main__":
    test_feedbacks = [
        "This is confusing, I need help.",
        "Great work, I love it!",
        "Where do I find my application ID?",
        "The page is not loading.",
        "The website is slow and I think it is just ok for my needs.",
    ]
    
    for feedback in test_feedbacks:
        print(f"Feedback: '{feedback}' → Classified as: {classify_feedback(feedback)}")
