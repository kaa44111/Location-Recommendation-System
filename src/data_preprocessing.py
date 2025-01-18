import pandas as pd
from haversine import haversine, Unit


def load_data(filepath):
    """Load the raw dataset."""
    # Load dataset
    data = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        encoding='ISO-8859-1',
        names=['User_ID', 'Venue_ID', 'Venue_Category_ID', 'Category_Name',
            'Latitude', 'Longitude', 'Timezone_Offset', 'UTC_Time'],
        dtype={
            'User_ID': 'str',
            'Venue_ID': 'str',
            'Venue_Category_ID': 'str',
            'Category_Name': 'category',
            'Latitude': 'float32',
            'Longitude': 'float32',
            'Timezone_Offset': 'int16',
            'UTC_Time': 'str'
        }
    )

    return data

def preprocess_data(data):
    """Clean and preprocess the dataset."""
    # Remove duplicates
    data = data.drop_duplicates()
    data.reset_index(drop=True, inplace=True)

    # Handle missing values
    data = data.dropna()

    #------------------------
    # Convert UTC time
    data['UTC_Time'] = pd.to_datetime(data['UTC_Time'], format="%a %b %d %H:%M:%S %z %Y", errors='coerce')
    data = data.dropna(subset=['UTC_Time'])

    # Add timezone and local time
    data['Timezone_Offset'] = pd.to_timedelta(data['Timezone_Offset'], unit='m')
    data['Local_Time'] = data['UTC_Time'] + data['Timezone_Offset']

    # Remove UTC label after applying the offset
    data['Local_Time'] = data['Local_Time'].dt.tz_convert(None)

    # Drop unnecessary columns
    data = data.drop(columns=['UTC_Time', 'Timezone_Offset'])
    
    return data

def feature_engineering(data, categories_path):
    """Add engineered features like time buckets, user profiles, etc."""

    #print('Add Broader Categories')
    # Load category mapping
    category_table = pd.read_csv(categories_path)

    # Add missing categories dynamically if needed
    missing_category = pd.DataFrame({
        'Category ID': ['4e51a0c0bd41d3446defbb2e'],
        'Category Name': ['Ferry'],
        'Category Label': ['Travel and Transportation > Ferry']
    })
    category_table = pd.concat([category_table, missing_category], ignore_index=True)

    # Rename and merge
    category_table.rename(columns={'Category ID': 'Venue_Category_ID'}, inplace=True)
    data = data.merge(category_table, on='Venue_Category_ID', how='left')

    # Extract broader categories
    data['Broader_Category'] = data['Category Label'].str.split(' > ').str[0]

    # Keep only the relevant columns
    data = data.drop(columns=['Category Label','Category Name'])

    #------------------------
    #print("Derive Temporal Features")

    # Extract day of the week
    data['Day_of_Week'] = data['Local_Time'].dt.day_name()

    # Identify if the visit was on a weekend
    data['Is_Weekend'] = data['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

    # Extract hour to categorize the time of visit
    data['Hour'] = data['Local_Time'].dt.hour

    # Create time buckets (e.g., Morning, Afternoon, Evening, Night)
    def time_bucket(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    data['Time_Bucket'] = data['Hour'].apply(time_bucket)

    #------------------------------
    #print("Create User Profiles")
    # Most visited category for each user
    user_top_category = (
        data.groupby(['User_ID', 'Category_Name'])
        .size()
        .reset_index(name='Visit_Count')
        .sort_values(['User_ID', 'Visit_Count'], ascending=[True, False])
        .drop_duplicates('User_ID')
    )
    data = data.merge(user_top_category[['User_ID', 'Category_Name']], on='User_ID', how='left', suffixes=('', '_Preferred'))

    # Most frequent time bucket for each user
    user_top_time = (
        data.groupby(['User_ID', 'Time_Bucket'])
        .size()
        .reset_index(name='Visit_Count')
        .sort_values(['User_ID', 'Visit_Count'], ascending=[True, False])
        .drop_duplicates('User_ID')
    )
    data = data.merge(user_top_time[['User_ID', 'Time_Bucket']], on='User_ID', how='left', suffixes=('', '_Preferred'))
    #---------------------------------
    #print("Compute venue popularity")

    # Compute venue popularity
    venue_popularity = data.groupby('Venue_ID')['User_ID'].count().reset_index(name='totalVisits')

    # Normalize popularity
    venue_popularity['Popularity_Score'] = venue_popularity['totalVisits'] / venue_popularity['totalVisits'].max()

    # Merge popularity back into the main dataset
    data = data.merge(venue_popularity[['Venue_ID', 'Popularity_Score']], on='Venue_ID', how='left')
    data = data.merge(venue_popularity[['Venue_ID', 'totalVisits']], on='Venue_ID', how='left')

    #-----------------------------
        # Count visits per venue and time bucket
    venue_time_bucket_visits = data.groupby(['Venue_ID', 'Time_Bucket']).size().reset_index(name='Visit_Count')

    # Identify the most popular time bucket for each venue
    venue_top_time_bucket = venue_time_bucket_visits.sort_values(['Venue_ID', 'Visit_Count'], ascending=[True, False])\
        .drop_duplicates('Venue_ID')\
        .rename(columns={'Time_Bucket': 'Busy_TimeBucket', 'Visit_Count': 'Max_Visit_Count'})

    # Merge the most popular time bucket back into the main dataset
    data = data.merge(venue_top_time_bucket[['Venue_ID', 'Busy_TimeBucket']], on='Venue_ID', how='left')
    #print(data[['Venue_ID', 'Category_Name', 'Busy_TimeBucket']].head(10))

    #--------------------------------
    #print("Compute Geographic Features")

    # Compute user's average latitude and longitude
    user_location_center = data.groupby('User_ID')[['Latitude', 'Longitude']].mean().reset_index()
    user_location_center.rename(columns={'Latitude': 'Avg_Latitude', 'Longitude': 'Avg_Longitude'}, inplace=True)

    # Merge user's central location into the main dataset
    data = data.merge(user_location_center, on='User_ID', how='left')

    # Compute distance from the user's central location
    data['Distance_From_Center'] = data.apply(
        lambda row: haversine(
            (row['Avg_Latitude'], row['Avg_Longitude']),
            (row['Latitude'], row['Longitude']),
            unit=Unit.KILOMETERS
        ), axis=1
    )

    return data