import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku

# Load the datasets
display_data = pd.read_json('./json/cleaned_osm_data_described.json')  # Data to display
training_data = pd.read_json('./json/training_tourism_spots_extended.json')  # Training data generated earlier

# Prepare the training dataset
training_data['combined_text'] = training_data['place_name'] + " " + training_data['description']

# Encode genres as numerical labels
label_encoder = LabelEncoder()
training_data['genre_label'] = label_encoder.fit_transform(training_data['genre'])
num_classes = len(label_encoder.classes_)

# Tokenizer initialization
tokenizer = Tokenizer()
corpus = training_data['combined_text'].tolist()

# Fit tokenizer on the combined text corpus
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # Add 1 because tokenizer index starts from 1

# Create input sequences for training
input_sequences = tokenizer.texts_to_sequences(corpus)
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Prepare predictors (X) and labels (y)
X = input_sequences
y = ku.to_categorical(training_data['genre_label'], num_classes=num_classes)

# Define the genre prediction model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len))  # Embedding layer
model.add(Bidirectional(LSTM(150, return_sequences=True)))  # LSTM layer
model.add(Dropout(0.2))  # Dropout layer
model.add(LSTM(100))  # Another LSTM layer
model.add(Dense(128, activation='relu'))  # Dense layer
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=64, verbose=1)

# Plot training accuracy and loss
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()

# Function for predicting the genre from user input
def predict_genre(user_description):
    """
    Predict the genre based on user input.

    Args:
        user_description (str): User's input description.

    Returns:
        str: Predicted genre.
    """
    token_list = tokenizer.texts_to_sequences([user_description])
    token_list = pad_sequences(token_list, maxlen=max_sequence_len, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)
    predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_genre

# Prepare the display dataset for recommendations
display_data['combined_text'] = display_data['name'] + " " + display_data['description'] + " " + display_data['genre']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(display_data['combined_text'])

# Function for search recommendations
def search_recommendations(query, top_n=5):
    """
    Search for tourism spots similar to the user query.
    
    Args:
        query (str): User's search input.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: Top N recommended tourism spots.
    """
    # Preprocess and vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top N results
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = display_data.iloc[top_indices]
    recommendations['similarity'] = similarity_scores[top_indices]
    return recommendations[['name', 'description', 'genre', 'rating', 'similarity']]

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a description of your preference: ")
    predicted_genre = predict_genre(user_input)
    print(f"Predicted Genre: {predicted_genre}")
    recommendations = search_recommendations(predicted_genre)
    print("Top Recommendations:")
    print(recommendations)