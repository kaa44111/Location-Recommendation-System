import pandas as pd

def save_to_csv(data, filepath):
    """Save data to a CSV file."""
    data.to_csv(filepath, index=False)

def load_from_csv(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)