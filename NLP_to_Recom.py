import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku

# Load the datasets
display_data = pd.read_csv('./tourism_spots.csv')  # Data to display
training_data = pd.read_csv('./training_tourism_spots.csv')  # Data for training

# Prepare the training dataset
training_data['combined_text'] = training_data['title'] + " " + training_data['description'] + " " + training_data['genre']

# Tokenizer initialization
tokenizer = Tokenizer()
corpus = training_data['combined_text'].tolist()

# Fit tokenizer on the combined text corpus
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # Add 1 because tokenizer index starts from 1

# Create input sequences (n-gram sequences) for training
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Prepare predictors (X) and labels (y)
X = input_sequences[:, :-1]  # All but the last token
y = input_sequences[:, -1]   # Last token is the target
y = ku.to_categorical(y, num_classes=total_words)  # Convert to one-hot encoding

# Define the text generation model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))  # Embedding layer
model.add(Bidirectional(LSTM(150, return_sequences=True)))  # LSTM layer
model.add(Dropout(0.2))  # Dropout layer
model.add(LSTM(100))  # Another LSTM layer
model.add(Dense(128, activation='relu'))  # Dense layer
model.add(Dense(total_words, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=500, batch_size=64, verbose=1)

# Plot training accuracy and loss
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()

# Function for predicting the next word in a sequence
def predict_next_words(input_text, num_words=5):
    """
    Predict the next word(s) based on input text.

    Args:
        input_text (str): Initial text to complete.
        num_words (int): Number of words to predict.

    Returns:
        str: The completed sequence.
    """
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        output_word = tokenizer.index_word.get(predicted_index, '')
        input_text += ' ' + output_word
    return input_text

# Prepare the display dataset for recommendations
display_data['combined_text'] = display_data['title'] + " " + display_data['description'] + " " + display_data['genre']

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
    return recommendations[['title', 'description', 'genre', 'rating', 'similarity']]

# Function for recommendations based on genre prediction
def recommend_based_on_genre(user_description, top_n=5):
    """
    Predict the genre from user description and recommend places.
    
    Args:
        user_description (str): User's input description.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: Top N recommended places.
    """
    predicted_genre = predict_next_words(user_description, num_words=1)
    print(f"Predicted Genre: {predicted_genre.strip()}")
    return search_recommendations(predicted_genre, top_n=top_n)

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a description of your preference: ")
    recommendations = recommend_based_on_genre(user_input)
    print("Top Recommendations:")
    print(recommendations)