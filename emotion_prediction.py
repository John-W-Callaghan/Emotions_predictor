import pandas as pd
import matplotlib.pyplot as plt
from tokenise import tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk


# Load the dataset
x = pd.read_csv("text.csv")


# Tokenize and preprocess each sentence in the dataset
preprocessed_sentences = []
for index, row in x.iterrows():
    sentence = row['text']
    emotion = row['label']

    # Preprocess the sentence
    preprocessed_sentence = tokenize(sentence)

    # Store the preprocessed sentence along with the emotion label
    preprocessed_sentences.append((preprocessed_sentence, emotion))

# Convert the preprocessed data into a DataFrame
preprocessed_x = pd.DataFrame(preprocessed_sentences, columns=['preprocessed_sentence', 'label'])



# Split the data into features (X) and labels (y)
X = x['text']
y = x['label']

# Mapping of numerical labels to emotion labels
emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_vectorized, y_train)

# Predict the labels for the test set
y_pred = logreg_model.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, y_pred)

# Define a function to predict emotions

def predict_emotion(sentence):
    sentence_vectorized = vectorizer.transform([sentence])
    predicted_label = logreg_model.predict(sentence_vectorized)
    predicted_emotion = emotion_mapping[predicted_label[0]]
    return predicted_emotion

# Define function to handle button click event
def predict_button_click():
    sentence = entry.get()
    predicted_emotion = predict_emotion(sentence)
    result_label.config(text="Predicted emotion: " + predicted_emotion)

# Create main window
root = tk.Tk()
root.title("Emotion Prediction")

# Create label and entry widgets
label = ttk.Label(root, text="Enter a sentence:")
label.grid(row=0, column=0, padx=5, pady=5)
entry = ttk.Entry(root, width=50)
entry.grid(row=0, column=1, padx=5, pady=5)

# Create predict button
predict_button = ttk.Button(root, text="Predict Emotion", command=predict_button_click)
predict_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

# Create label to display result
result_label = ttk.Label(root, text="")
result_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Run the application
root.mainloop()