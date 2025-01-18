import sys
import os

import sys
print(sys.path)
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import pytest
from src.recommendation_unvisisted import recommend_similar_category_locations


def test_recommend_similar_category_locations():
    # Mock dataset and inputs
    data = pd.DataFrame({
        'User_ID': ['1', '1', '2', '3'],
        'Venue_ID': ['49bbd6c0f964a520f4531fe3', '4a43c0aef964a520c6a61fe3', '4c5cc7b485a1e21e00d35711', '4bc7086715a7ef3bef9878da'],
        'Category_Name': ['Bar', 'Cafe', 'Bar', 'Restaurant'],
        'Popularity_Score': [0.8, 0.6, 0.9, 0.5],
        'Latitude': [40.719810, 40.606800, 40.716160, 40.745163],
        'Longitude': [-74.002579, -74.044167, -73.883072, -73.982521],
        'Broader_Category' : ['Arts and Entertainment','Business and Professional Services',
                              'Community and Government', 'Dining and Drinking']
    })
    user_id = "1"
    category_name = "Bar"

    recommendations = recommend_similar_category_locations(user_id, category_name, data, top_k=2)
    assert len(recommendations) <= 2
    assert "Venue_ID" in recommendations.columns


from src.recommendation_point import recommend_meeting_place_random_checkins

def test_recommend_meeting_place():
    mock_data = pd.DataFrame({
        'User_ID': ['1', '2', '3', '4', '5'],
        'Venue_ID': ['49bbd6c0f964a520f4531fe3', 'B', 'C', 'D', 'E'],
        'Latitude': [40.7128, 40.7138, 40.7148, 40.7158, 40.7168],
        'Longitude': [-74.0060, -74.0070, -74.0080, -74.0090, -74.0100]
    })
    user_ids = ['1', '2', '3','4']
    selected_checkins, nearest_venues = recommend_meeting_place_random_checkins(user_ids, mock_data, k=1)
    
    # Assertions
    assert len(selected_checkins) == len(user_ids), "Incorrect number of user check-ins selected."
    assert len(nearest_venues) == 1, "Nearest venue calculation failed."
