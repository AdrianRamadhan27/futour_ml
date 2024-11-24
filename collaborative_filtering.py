import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model

# Dataset
data = pd.DataFrame({
    'user_id': [0, 1, 0, 2, 2],
    'tourism_spot_id': [0, 0, 1, 1, 2],
    'rating': [4, 5, 5, 3, 4]
})

# Data Preprocessing
user_encoder = LabelEncoder()
spot_encoder = LabelEncoder()

data['user_id'] = user_encoder.fit_transform(data['user_id'])
data['tourism_spot_id'] = spot_encoder.fit_transform(data['tourism_spot_id'])

train, test = train_test_split(data, test_size=0.2, random_state=42)

train_user = train['user_id'].values
train_spot = train['tourism_spot_id'].values
train_ratings = train['rating'].values

test_user = test['user_id'].values
test_spot = test['tourism_spot_id'].values
test_ratings = test['rating'].values

# Defining TensorFlow Model
num_users = data['user_id'].nunique()
num_spots = data['tourism_spot_id'].nunique()
embedding_dim = 50

user_input = layers.Input(shape=(1,), name='user_input')
user_embedding = layers.Embedding(num_users, embedding_dim, name='user_embedding')(user_input)
user_vec = layers.Flatten()(user_embedding)

spot_input = layers.Input(shape=(1,), name='spot_input')
spot_embedding = layers.Embedding(num_spots, embedding_dim, name='spot_embedding')(spot_input)
spot_vec = layers.Flatten()(spot_embedding)

dot_product = layers.Dot(axes=1)([user_vec, spot_vec])

model = Model(inputs=[user_input, spot_input], outputs=dot_product)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(
    [train_user, train_spot], train_ratings,
    validation_data=([test_user, test_spot], test_ratings),
    epochs=10, batch_size=32, verbose=1
)

# Make Predictions for Top Recommendations
def get_top_n_recommendations(user_id, model, spot_encoder, data, n=5):
    """
    Get top N recommendations for a given user.
    
    Args:
        user_id (int): The ID of the user to recommend for.
        model (Model): The trained collaborative filtering model.
        spot_encoder (LabelEncoder): Encoder for tourism spot IDs.
        data (pd.DataFrame): Original dataset containing user-spot interactions.
        n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: Top N recommended spots with predicted ratings.
    """
    # Get all unique tourism spots
    all_spots = set(data['tourism_spot_id'].unique())
    
    # Get spots the user has already interacted with
    interacted_spots = set(data[data['user_id'] == user_id]['tourism_spot_id'])
    
    # Get spots the user hasn't interacted with
    unseen_spots = list(all_spots - interacted_spots)
    
    # Predict ratings for unseen spots
    user_array = np.array([user_id] * len(unseen_spots))
    spot_array = np.array(unseen_spots)
    predicted_ratings = model.predict([user_array, spot_array])
    
    # Combine predictions with spot IDs
    recommendations = pd.DataFrame({
        'tourism_spot_id': spot_array,
        'predicted_rating': predicted_ratings.flatten()
    })
    
    # Sort by predicted rating and return top N
    top_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(n)
    
    # Decode spot IDs back to their original form
    top_recommendations['tourism_spot_id'] = spot_encoder.inverse_transform(top_recommendations['tourism_spot_id'])
    
    return top_recommendations

# Example usage
user_id = 0
top_n_recommendations = get_top_n_recommendations(user_id, model, spot_encoder, data, n=3)
print(f"Top Recommendations for user {user_id}:")
print(top_n_recommendations)

# Step 6: Save the Model
model.save('collaborative_filtering_model.h5')
print("Model saved successfully!")