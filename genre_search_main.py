from genre_model import GenreModel
from search_recommendation import SearchRecommendations
import pandas as pd

# Load datasets
training_data = pd.read_json('./json/training_dataset_1.json')
display_data = pd.read_json('./json/cleaned_osm_data_described.json')

# Initialize the genre model (will load saved model if available)
genre_model = GenreModel(training_data)

# Initialize the search recommendation system
search_recs = SearchRecommendations(display_data)

# User input
user_input = input("Enter a description of your preference: ")
# predicted_genre = genre_model.predict_genre(user_input)
# print(f"Predicted Genre: {predicted_genre}")

# Recommendations based on predicted genre
predicted_recommendations = genre_model.predict_next_words(user_input)
genre_prediction = search_recs.return_genre(predicted_recommendations)
recommendations = search_recs.search_recommendations(predicted_recommendations, top_n=5)
print(f"Predicted Genre: {genre_prediction}")
print("Top Recommendations:")
print(recommendations)
