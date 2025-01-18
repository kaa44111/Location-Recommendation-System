import sys
import os

import sys
print(sys.path)
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Import functions from your implementation
from src.similarity import compute_user_profile, compute_user_similarity, find_top_similar_users

@pytest.fixture
def mock_data():
    """Fixture to create mock data for testing."""
    return pd.DataFrame({
        'User_ID': ['U1', 'U2', 'U3', 'U4'],  # Ensure these IDs match the similarity matrix
        'Category_Name_Preferred': ['Bar', 'Cafe', 'Bar', 'Restaurant'],
        'Time_Bucket_Preferred': ['Evening', 'Morning', 'Evening', 'Afternoon'],
        'Avg_Latitude': [40.7128, 40.7306, 40.7158, 40.7290],
        'Avg_Longitude': [-74.0060, -73.9352, -74.0020, -73.9910]
    })


# Test for compute_user_profile
def test_compute_user_profile(mock_data):
    user_profiles = compute_user_profile(mock_data)

    # Assert that the result is a DataFrame
    assert isinstance(user_profiles, pd.DataFrame), "Output is not a DataFrame."

    # Assert that User_ID is included
    assert 'User_ID' in user_profiles.columns, "User_ID column is missing in the output."

    # Assert that one-hot encoded and normalized features are included
    assert 'Category_Name_Preferred_Bar' in user_profiles.columns, "One-hot encoding for Category_Name_Preferred failed."
    assert 'Avg_Latitude' in user_profiles.columns, "Avg_Latitude normalization failed."
    assert 'Avg_Longitude' in user_profiles.columns, "Avg_Longitude normalization failed."

# Test for compute_user_similarity
def test_compute_user_similarity(mock_data):
    user_profiles = compute_user_profile(mock_data)
    user_similarity_df = compute_user_similarity(user_profiles)

    # Assert that the result is a DataFrame
    assert isinstance(user_similarity_df, pd.DataFrame), "Output is not a DataFrame."

    # Assert that the DataFrame is square
    assert user_similarity_df.shape[0] == user_similarity_df.shape[1], "Similarity matrix is not square."

    # Assert that diagonal values are 1
    for user in user_similarity_df.index:
        assert user_similarity_df.loc[user, user] == pytest.approx(1.0), f"Diagonal value for {user} is not 1."

# Test for find_top_similar_users
def test_find_top_similar_users(mock_data):
    user_profiles = compute_user_profile(mock_data)
    user_similarity_df = compute_user_similarity(user_profiles)

    # Test with a valid user
    top_similar_users = find_top_similar_users('U1', user_similarity_df, top_n=2)

    # Assert that the output is a Series
    assert isinstance(top_similar_users, pd.Series), "Output is not a Series."

    # Assert that the correct number of users is returned
    assert len(top_similar_users) == 2, "Incorrect number of similar users returned."

    # Assert that the user ID is excluded from the results
    assert 'U1' not in top_similar_users.index, "Self-similarity is included in the results."

    # Test with an invalid user
    with pytest.raises(ValueError, match="User ID U999 not found in the dataset."):
        find_top_similar_users('U999', user_similarity_df)

# Edge case tests
def test_empty_data():
    empty_data = pd.DataFrame(columns=['User_ID', 'Category_Name_Preferred', 'Time_Bucket_Preferred', 'Avg_Latitude', 'Avg_Longitude'])
    user_profiles = compute_user_profile(empty_data)

    # Assert that the output is empty
    assert user_profiles.empty, "User profiles for empty data should be empty."

    # Create an empty similarity DataFrame for consistency
    empty_similarity_df = pd.DataFrame()

    with pytest.raises(ValueError, match="User ID U1 not found in the dataset."):
        find_top_similar_users('U1', empty_similarity_df)


# Run tests using pytest
def main():
    pytest.main(["-v", __file__])

if __name__ == "__main__":
    main()
