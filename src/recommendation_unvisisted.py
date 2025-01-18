import pandas as pd

def recommend_similar_category_locations(user_id, category_name, data, top_k=10):
    """
    Recommend unique venues of a similar category for a user.

    Args:
        user_id (str): User ID.
        category_name (str): The specific venue category to find similar categories.
        data (pd.DataFrame): Dataset with user and venue information.
        top_k (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: Top recommended venues with scores.
    """
    # Normalize input category name
    category_name = category_name.lower()
    
    # Get the broader category for the input category
    try:
        broader_category = data.loc[
            data['Category_Name'].str.lower() == category_name, 'Broader_Category'
        ].values[0]
    except IndexError:
        raise ValueError(f"Category name '{category_name}' not found in the dataset.")
    
    # Filter data for venues in the broader category
    similar_venues = data[data['Broader_Category'] == broader_category]

    # Drop duplicate venues
    similar_venues = similar_venues.drop_duplicates(subset='Venue_ID')

    # Exclude venues already visited by the user
    visited = set(data[data['User_ID'] == user_id]['Venue_ID'])
    unvisited = similar_venues[~similar_venues['Venue_ID'].isin(visited)].copy()
    
    if unvisited.empty:
        return pd.DataFrame(columns=['Venue_ID', 'Category_Name', 'Score'])
    
    # Calculate scores based on popularity and proximity
    unvisited['Score'] = unvisited['Popularity_Score'] / (1 + unvisited['Distance_From_Center'])
    
    # Return the top-k unique venues
    return unvisited.nlargest(top_k, 'Score')[['Venue_ID', 'Category_Name', 'Score', 'Latitude', 'Longitude']]
