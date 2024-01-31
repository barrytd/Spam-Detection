from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset (replace this with your own dataset)
messages = [
    ("Free Viagra now!!!", "spam"),
    ("Hey, how are you?", "ham"),
    ("Win a free iPhone", "spam"),
    ("Meeting at 3 PM", "ham"),
    ("Claim your prize!", "suspicious"),
    ("You've won!", "spam")
    # Add more examples...
]

# Separate the messages and labels
texts, labels = zip(*messages)

# Convert text data to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions, zero_division=1)  # or zero_division=0

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

