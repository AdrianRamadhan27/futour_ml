# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('tourism_spots.csv')

# Combine relevant fields into one text column
data['combined_text'] = data['title'] + " " + data['description'] + " " + data['genre']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

# Function for basic search recommendations
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
    recommendations = data.iloc[top_indices]
    recommendations['similarity'] = similarity_scores[top_indices]
    return recommendations[['title', 'description', 'genre', 'rating', 'similarity']]

# Function for search recommendations with rating-based ranking
def search_recommendations_with_rating(query, top_n=5, rating_weight=0.3):
    """
    Search and rank recommendations using similarity and rating.
    
    Args:
        query (str): User's search input.
        top_n (int): Number of recommendations to return.
        rating_weight (float): Weight for the normalized rating.
    
    Returns:
        pd.DataFrame: Top N recommended tourism spots.
    """
    # Preprocess and vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Normalize ratings
    normalized_ratings = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())
    
    # Calculate weighted score
    weighted_score = similarity_scores * (1 - rating_weight) + normalized_ratings * rating_weight
    
    # Get top N results
    top_indices = weighted_score.argsort()[-top_n:][::-1]
    recommendations = data.iloc[top_indices]
    recommendations['similarity'] = similarity_scores[top_indices]
    recommendations['weighted_score'] = weighted_score[top_indices]
    return recommendations[['title', 'description', 'genre', 'rating', 'similarity', 'weighted_score']]

# Example usage
if __name__ == "__main__":
    query = input("Enter your search query: ")
    print("\nTop recommendations based on similarity:")
    print(search_recommendations(query, top_n=5))
    
    print("\nTop recommendations with similarity and rating:")
    print(search_recommendations_with_rating(query, top_n=5, rating_weight=0.4))
