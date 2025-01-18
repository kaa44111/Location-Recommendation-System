import folium

def visualize_random_checkins_and_venues(selected_checkins, nearest_venues):
    """
    Visualize the selected user check-ins and recommended venues on a map.

    Args:
        selected_checkins (pd.DataFrame): DataFrame containing randomly selected user check-ins with:
            - User_ID: Unique identifier for the user.
            - Latitude: Latitude of the check-in.
            - Longitude: Longitude of the check-in.
        nearest_venues (pd.DataFrame): DataFrame containing recommended venues with:
            - Venue_ID: Unique identifier for the venue.
            - Category_Name: Name of the venue's category.
            - Latitude: Latitude of the venue.
            - Longitude: Longitude of the venue.
            - Distance_From_Central (optional): Distance from the central meeting point.

    Returns:
        folium.Map: A map showing both user check-ins and recommended venues.
    """
    # Use the first user's check-in as the map's center
    center_lat = selected_checkins['Latitude'].iloc[0]
    center_long = selected_checkins['Longitude'].iloc[0]

    # Initialize the map
    combined_map = folium.Map(location=[center_lat, center_long], zoom_start=13)

    # Add user check-in markers
    for _, row in selected_checkins.iterrows():
        user_id = row['User_ID']
        lat = row['Latitude']
        long = row['Longitude']

        popup = folium.Popup(f"User ID: {user_id}<br>Latitude: {lat:.6f}<br>Longitude: {long:.6f}", max_width=300)
        folium.Marker(
            location=[lat, long],
            popup=popup,
            icon=folium.Icon(color='blue', icon='user', prefix='fa')  # Blue marker for users
        ).add_to(combined_map)

    # Add venue markers
    for _, row in nearest_venues.iterrows():
        venue_id = row['Venue_ID']
        category = row['Category_Name']
        lat = row['Latitude']
        long = row['Longitude']
        distance = row.get('Distance_From_Central', None)

        # Create popup with venue details
        details = f"Venue ID: {venue_id}<br>Category: {category}"
        if distance is not None:
            details += f"<br>Distance from Central: {distance:.2f} km"

        popup = folium.Popup(details, max_width=300)
        folium.Marker(
            location=[lat, long],
            popup=popup,
            icon=folium.Icon(color='red', icon='cutlery', prefix='fa')  # Red marker for venues
        ).add_to(combined_map)

    return combined_map

