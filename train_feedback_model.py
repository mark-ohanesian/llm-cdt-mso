import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample training data (you can expand this later)
feedback_samples = [
    "Great website, very helpful!",  # Kudos
    "I can't find the information I need.",  # Problem
    "How do I apply for a permit?",  # Question
    "This page is broken, please fix it.",  # Problem
    "Excellent service, thanks!",  # Kudos
    "Who should I contact for support?",  # Question
]

categories = ["kudos", "problem", "question", "problem", "kudos", "question"]

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
    test_feedback = "I need help with my application."
    print(f"Feedback: {test_feedback}\nClassified as: {classify_feedback(test_feedback)}")
