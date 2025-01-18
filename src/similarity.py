import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def compute_user_profile(data):
    """Create a User Profile with relevant features."""
    if data.empty:
        return pd.DataFrame(columns=['User_ID'])  # Return an empty DataFrame if input is empty

    # Extract relevant user features
    user_features = data[['User_ID', 'Category_Name_Preferred', 'Time_Bucket_Preferred', 'Avg_Latitude', 'Avg_Longitude']].drop_duplicates()

    # One-hot encode categorical features
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(user_features[['Category_Name_Preferred', 'Time_Bucket_Preferred']])

    # Convert encoded features to DataFrame
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out())

    # Normalize numerical features
    scaler = MinMaxScaler()
    normalized_coords = scaler.fit_transform(user_features[['Avg_Latitude', 'Avg_Longitude']])
    normalized_coords_df = pd.DataFrame(normalized_coords, columns=['Avg_Latitude', 'Avg_Longitude'])

    # Combine all features
    user_features_combined = pd.concat(
        [user_features[['User_ID']].reset_index(drop=True), encoded_df, normalized_coords_df], axis=1
    )

    return user_features_combined


def compute_user_similarity(user_data):
    """Compute similarity between users."""

    # Set User_ID as the index
    user_features_matrix = user_data.set_index('User_ID')

    # Compute cosine similarity
    user_similarity = cosine_similarity(user_features_matrix)

    # Convert similarity matrix to DataFrame
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_features_matrix.index,
        columns=user_features_matrix.index
    )

    return user_similarity_df

def find_top_similar_users(user_id, user_similarity_df, top_n=10):
    """
    Find the top N most similar users for a given user.

    Args:
        user_id (str): The user ID to find similar users for.
        user_similarity_df (pd.DataFrame): User similarity matrix.
        top_n (int): Number of similar users to return.

    Returns:
        pd.Series: Top N similar users and their similarity scores.
    """
    if user_id not in user_similarity_df.index:
        raise ValueError(f"User ID {user_id} not found in the dataset.")
    
    # Sort similar users by similarity score, excluding the user themselves
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).iloc[1:top_n + 1]
    return similar_users

if __name__ == "__main__":
    # Load the preprocessed data
    data = pd.read_csv("data/processed_data.csv", sep="\t", encoding="ISO-8859-1")

    # Precompute user similarity matrix
    user_data = compute_user_profile(data)
    user_similarity_df = compute_user_similarity(user_data)

    print(user_similarity_df)