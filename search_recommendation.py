import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SearchRecommendations:
    def __init__(self, display_data):
        self.display_data = display_data
        self.display_data['combined_text'] = (
            self.display_data['name'] + " " + 
            self.display_data['description'] + " " + 
            self.display_data['genre']
        )
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.display_data['combined_text'])

    def search_recommendations(self, query, top_n=5):
        """
        Search for recommendations based on a query or predicted genre.

        Args:
            query (str): User's search input or predicted genre.
            top_n (int): Number of recommendations to return.

        Returns:
            pd.DataFrame: Top N recommended tourism spots.
        """
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommendations = self.display_data.iloc[top_indices]
        recommendations['similarity'] = similarity_scores[top_indices]
        return recommendations[['name', 'description', 'genre', 'rating', 'similarity']]
    
    def return_genre(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_index = similarity_scores.argmax()  # Get the index of the highest similarity score
        
        # Return only the genre of the top recommendation
        top_genre = self.display_data.iloc[top_index]['genre']
        return top_genre