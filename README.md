# README: Recommendation System for Location-Based Recommendations

## Overview
This project implements a recommendation system that suggests venues based on user preferences, computes similar users, and recommends meeting places. It includes data preprocessing, a recommendation module, similarity computations, and visualization tools. A lightweight GUI is also provided for user interaction.

## Prerequisites
- Python 3.8+
- Install dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Final_assignment
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
Final_assignment/
├── src/                       # Source code
│   ├── __init__.py            # Makes src a package
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── recommendation.py      # Recommendation functions
│   ├── similarity.py          # Similarity computations
│   ├── visualization.py       # Folium visualization
│   └── utils.py               # Helper utilities
│
├── tests/                     # Unit and integration tests
│   ├── test_recommendation.py
│   ├── test_data_preprocessing.py
│   ├── test_visualization.py
│   └── __init__.py
│
├── data/                      # Datasets
│   ├── dataset_NYC.txt
│   └── categories.csv
│
├── notebooks/                 # Exploration Jupyter notebooks
│   └── recommendation.ipynb
│
├── README.txt                 # Instructions on how to run the project
├── requirements.txt           # Dependencies
├── setup.py                   # Makes project installable
├── main.py                    # Main script to tie everything together
└── outputs/                   # Output files (e.g., CSVs, screenshots)
    └── results.html
```

## How to Run
1. **Prepare the data:** Place the dataset files (`dataset_NYC.txt`, `categories.csv`) into the `data/` folder.
2. **Run the main script:**
   ```bash
   python main.py
   ```
   The main script performs data preprocessing, generates recommendations, and evaluates results.
3. **Access the GUI (Optional):**
   To launch the GUI for dynamic interaction, run:
   ```bash
   streamlit run src/gui.py
   ```

## Features
1. **Unvisited Location Recommendations:**
   - Input: `User_ID` and `Category_Name`
   - Output: Top 10 unvisited venues based on user preferences and category.

2. **Similar User Recommendations:**
   - Input: `User_ID`
   - Output: Top 10 users with similar preferences.

3. **Meeting Place Recommendations:**
   - Input: A list of `User_IDs`
   - Output: The nearest meeting place for the group.

4. **Visualization:**
   - Interactive Folium maps to display recommendations and user check-ins.

## Evaluation Metrics
- **Precision:** Measures relevance of recommended venues.
- **Diversity:** Ensures uniqueness of recommendations.
- **Proximity:** Calculates average distance for meeting place recommendations.

## Testing
Run the test suite to validate the project:
```bash
pytest
```

## Outputs
- Recommendations are displayed in the terminal and saved in the `outputs/` folder.
- Interactive maps are saved as HTML files (e.g., `recommendation_map.html`).

## Notes
- Ensure geospatial libraries (e.g., `geopy`) are installed and working.
- Use a virtual environment to manage dependencies.
