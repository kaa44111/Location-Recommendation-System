import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, feature_engineering
from src.similarity import compute_user_profile, compute_user_similarity, find_top_similar_users
from src.recommendation_unvisisted import recommend_similar_category_locations
from src.recommendation_point import recommend_meeting_place_random_checkins
from src.visualization import visualize_random_checkins_and_venues


# Main function to run all computations
def main():
    # Step 1: Load Data
    filepath = "data/dataset_NYC.txt"
    print("Loading data...")
    data = load_data(filepath)

    # Step 2: Preprocess and Feature Engineer
    #print("Preprocessing data...")
    data = preprocess_data(data)
    categories_path= "data/categories.csv"
    #print("Feature engineering...")
    data = feature_engineering(data,categories_path)

    # Step 3: Compute User Profiles
    #print("Computing user profiles...")
    user_profiles = compute_user_profile(data)

    # Step 4: Compute User Similarity
    #print("Computing user similarity...")
    user_similarity_df = compute_user_similarity(user_profiles)

    # Step 5: Recommendation - Unvisited Locations
    user_id = "20"
    category_name = "Bar"
    print(f"Recommending unvisited locations for User {user_id} in category {category_name}...")
    recommendations = recommend_similar_category_locations(user_id, category_name, data, top_k=10)
    print("Recommendations:")
    print(recommendations)

    # Step 6: Recommendation - Similar Users
    print(f"Finding top 10 similar users for User {user_id}...")
    similar_users = find_top_similar_users(user_id, user_similarity_df, top_n=10)
    print("Top similar users:")
    print(similar_users)

    # Step 7: Recommendation - Meeting Place
    user_ids=['470', '979', '69', '395', '87']
    print(f"Recommending meeting place for users {user_ids}...")
    selected_checkins, nearest_venues = recommend_meeting_place_random_checkins(user_ids, data, k=3)
    print("Recommended meeting place:")
    print(nearest_venues)

    # Step 8: Visualization
    #print("Visualizing meeting place...")
    #combined_map = visualize_random_checkins_and_venues(selected_checkins, nearest_venues)
    #combined_map.save("meeting_place_map.html")
    #print("Map saved to meeting_place_map.html")


if __name__ == "__main__":
    main()

