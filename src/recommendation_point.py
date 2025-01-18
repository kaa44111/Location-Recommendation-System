import random
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic

def select_random_checkins(user_ids, data):
    """
    Randomly select one check-in per user from the dataset.

    Args:
        user_ids (list): List of user IDs.
        data (pd.DataFrame): Dataset with user and check-in information.

    Returns:
        pd.DataFrame: Selected check-ins for the given users.
    """
    # Filter data for the specified user IDs
    user_checkins = data[data['User_ID'].isin(user_ids)]
    
    # Randomly select one check-in per user
    random_checkins = user_checkins.groupby('User_ID').apply(lambda x: x.sample(1)).reset_index(drop=True)
    
    return random_checkins[['User_ID', 'Latitude', 'Longitude']]

def get_central_meeting_point(selected_checkins):
    """
    Calculate the central meeting point for the selected check-ins.

    Args:
        selected_checkins (pd.DataFrame): Selected check-ins with Latitude and Longitude.

    Returns:
        tuple: Central latitude and longitude for the meeting point.
    """
    central_lat = selected_checkins['Latitude'].mean()
    central_lon = selected_checkins['Longitude'].mean()
    
    return central_lat, central_lon

def find_nearest_venues(central_point, data, k=1):
    """
    Use KNN to find the nearest venues to the central meeting point, ensuring unique Venue_IDs.

    Args:
        central_point (tuple): Central latitude and longitude.
        data (pd.DataFrame): Dataset with venue information.
        k (int): Number of nearest venues to return.

    Returns:
        pd.DataFrame: The k nearest unique venues.
    """
    # Ensure unique Venue_IDs
    data = data.drop_duplicates(subset='Venue_ID')

    # Prepare venue location data
    venue_locations = data[['Latitude', 'Longitude']].to_numpy()

    # Initialize KNN model
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(venue_locations)

    # Find the nearest venues
    distances, indices = knn.kneighbors([central_point])

    # Extract the nearest venues
    nearest_venues = data.iloc[indices[0]].copy()
    nearest_venues['Distance_From_Central'] = distances[0]

    return nearest_venues

def recommend_meeting_place_random_checkins(user_ids, data, k=1):
    """
    Recommend the nearest meeting place for a group of users by selecting random check-ins.

    Args:
        user_ids (list): List of user IDs.
        data (pd.DataFrame): Dataset with user and venue information.
        k (int): Number of nearest venues to return.

    Returns:
        tuple: The selected check-ins and the nearest venue(s).
    """
    # Step 1: Randomly select one check-in per user
    selected_checkins = select_random_checkins(user_ids, data)
    
    # Step 2: Calculate the central meeting point
    central_point = get_central_meeting_point(selected_checkins)
    
    # Step 3: Find the nearest venues
    nearest_venues = find_nearest_venues(central_point, data, k=k)
    
    return selected_checkins, nearest_venues

