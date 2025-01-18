import streamlit as st
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, feature_engineering
from src.recommendation_point import recommend_meeting_place_random_checkins
from src.recommendation_unvisisted import recommend_similar_category_locations
from src.similarity import compute_user_profile, compute_user_similarity, find_top_similar_users

# Load and preprocess data (cached for performance)
@st.cache_data
def load_and_prepare_data():
    raw_data = load_data("data/dataset_NYC.txt")
    cleaned_data = preprocess_data(raw_data)
    processed_data = feature_engineering(cleaned_data,'data/categories.csv')
    user_profiles = compute_user_profile(processed_data)
    user_similarity_df = compute_user_similarity(user_profiles)
    return processed_data, user_profiles, user_similarity_df

# Main Streamlit app
def main():
    st.title("Location Recommendation System")
    st.sidebar.title("Menu")
    
    # Load data
    processed_data, user_profiles, user_similarity_df = load_and_prepare_data()
    
    # Menu options
    option = st.sidebar.selectbox(
        "Choose a feature:",
        ("Recommend Unvisited Locations", "Find Similar Users", "Recommend Meeting Place")
    )

    if option == "Recommend Unvisited Locations":
        st.header("Recommend Unvisited Locations")
        user_id = st.text_input("Enter User ID (e.g., 20):", "20")
        category_name = st.text_input("Enter Category Name (e.g., Bar):", "Bar")
        top_k = st.slider("Number of Recommendations:", 1, 20, 10)

        if st.button("Get Recommendations"):
            recommendations = recommend_similar_category_locations(user_id, category_name, processed_data, top_k)
            st.write("Recommended Locations:")
            st.dataframe(recommendations)

    elif option == "Find Similar Users":
        st.header("Find Similar Users")
        user_id = st.text_input("Enter User ID (e.g., 20):", "20")
        top_n = st.slider("Number of Similar Users:", 1, 20, 10)

        if st.button("Find Similar Users"):
            similar_users = find_top_similar_users(user_id, user_similarity_df, top_n)
            st.write("Top Similar Users:")
            st.dataframe(similar_users)

    elif option == "Recommend Meeting Place":
        st.header("Recommend Meeting Place")
        user_ids = st.text_input("Enter User IDs (comma-separated, e.g., U1,U2,U3,U4,U5):", "U1,U2,U3,U4,U5")
        user_ids = [uid.strip() for uid in user_ids.split(",")]

        if st.button("Get Meeting Place"):
            selected_checkins, nearest_venues = recommend_meeting_place_random_checkins(user_ids, processed_data, k=1)
            st.write("Recommended Meeting Place:")
            st.dataframe(nearest_venues)

# Run the app
if __name__ == "__main__":
    main()
