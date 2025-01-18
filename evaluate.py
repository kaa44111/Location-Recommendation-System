import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, feature_engineering
from src.similarity import compute_user_profile, compute_user_similarity, find_top_similar_users
from src.recommendation_unvisisted import recommend_similar_category_locations
from src.recommendation_point import recommend_meeting_place_random_checkins

def load_and_prepare_data():
    raw_data = load_data("data/dataset_NYC.txt")
    cleaned_data = preprocess_data(raw_data)
    processed_data = feature_engineering(cleaned_data,'data/categories.csv')
    user_profiles = compute_user_profile(processed_data)
    user_similarity_df = compute_user_similarity(user_profiles)
    return processed_data, user_profiles, user_similarity_df

def evaluate_recommend_unvisited(data, user_id, category_name, top_k=10):
    # Get recommendations
    recommendations = recommend_similar_category_locations(user_id, category_name, data, top_k)

    # Precision: Check if recommendations match the input category
    precision = (recommendations['Category_Name'].str.lower() == category_name.lower()).mean()

    # Diversity: Ratio of unique venues
    diversity = recommendations['Venue_ID'].nunique() / len(recommendations)

    # Print metrics
    print(f"Precision: {precision:.2f}")
    print(f"Diversity: {diversity:.2f}")
    print(f"Recommended Locations:\n{recommendations[['Venue_ID', 'Category_Name', 'Score']]}")

def evaluate_user_similarity(data, user_id, user_similarity_df, top_n=10):
    # Get similar users
    similar_users = find_top_similar_users(user_id, user_similarity_df, top_n)

    # Average similarity score
    avg_similarity = similar_users.mean()

    # Calculate overlaps
    overlaps = []
    user_pref = data[data['User_ID'] == user_id][['Category_Name', 'Time_Bucket']].drop_duplicates()

    for similar_user in similar_users.index:
        similar_user_pref = data[data['User_ID'] == similar_user][['Category_Name', 'Time_Bucket']].drop_duplicates()

        # Calculate intersection of preferences
        overlap = pd.merge(user_pref, similar_user_pref, on=['Category_Name', 'Time_Bucket']).shape[0]
        overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

    # Print metrics
    print(f"Average Similarity Score: {avg_similarity:.2f}")
    print(f"Average Preference Overlap: {avg_overlap:.2f}")


from geopy.distance import geodesic

def evaluate_meeting_place(data, user_ids, k=3):
    """
    Evaluate the recommended meeting places for a group of users.

    Args:
        data (pd.DataFrame): The dataset containing user and venue information.
        user_ids (list): List of user IDs.
        k (int): Number of meeting places to evaluate.

    Returns:
        None
    """
    # Get recommended meeting places and user check-ins
    print(f"Recommending meeting place for users {user_ids}...")
    selected_checkins, nearest_venues = recommend_meeting_place_random_checkins(user_ids, data, k)

    # Calculate average distances for each venue
    avg_distances = []
    for _, venue in nearest_venues.iterrows():
        meeting_coords = (venue['Latitude'], venue['Longitude'])
        distances = [
            geodesic((row['Latitude'], row['Longitude']), meeting_coords).km
            for _, row in selected_checkins.iterrows()
        ]
        avg_distances.append(sum(distances) / len(distances))

    # Add average distances to the nearest venues DataFrame
    nearest_venues['Avg_Distance'] = avg_distances

    # Print results
    print(f"User Check-ins:\n{selected_checkins[['User_ID', 'Latitude', 'Longitude']]}\n")
    print(f"Recommended Meeting Places:\n{nearest_venues[['Venue_ID', 'Category_Name', 'Latitude', 'Longitude', 'Avg_Distance']]}")




# Main function to run all computations
def main():
    data, user_profiles, user_similarity_df = load_and_prepare_data()

    print('evaluate_recommend_unvisited')
    evaluate_recommend_unvisited(data, user_id="20", category_name="Bar", top_k=10)

    print('evaluate_user_similarity')
    user_profiles = compute_user_profile(data)
    user_similarity_df = compute_user_similarity(user_profiles)
    evaluate_user_similarity(data, user_id="20", user_similarity_df=user_similarity_df, top_n=10)

    evaluate_meeting_place(data, user_ids=['470', '979', '69', '395', '87'], k=1)


if __name__ == "__main__":
    main()

