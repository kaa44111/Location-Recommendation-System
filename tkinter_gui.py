import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from src.recommendation_point import recommend_meeting_place_random_checkins
from src.recommendation_unvisisted import recommend_similar_category_locations
from src.similarity import find_top_similar_users, compute_user_similarity, compute_user_profile
from src.data_preprocessing import load_data, preprocess_data, feature_engineering

def create_gui(data, user_similarity_df):
    root = tk.Tk()
    root.title("Recommendation System")
    root.geometry("800x600")

    tab_control = ttk.Notebook(root)
    tab_recommend = ttk.Frame(tab_control)
    tab_similar = ttk.Frame(tab_control)
    tab_meeting = ttk.Frame(tab_control)
    tab_control.add(tab_recommend, text="Unvisited Locations")
    tab_control.add(tab_similar, text="Similar Users")
    tab_control.add(tab_meeting, text="Meeting Place")
    tab_control.pack(expand=1, fill="both")

    # Tab 1: Recommend Unvisited Locations
    def handle_recommend():
        user_id = user_id_entry.get()
        category_name = category_entry.get()
        if not user_id or not category_name:
            messagebox.showerror("Error", "Please enter both User ID and Category Name")
            return

        try:
            results = recommend_similar_category_locations(user_id, category_name, data, top_k=10)
            if not isinstance(results, pd.DataFrame) or results.empty:
                messagebox.showerror("Error", "No recommendations found.")
                return

            recommendation_output.delete(*recommendation_output.get_children())
            for _, row in results.iterrows():
                recommendation_output.insert("", "end", values=(row["Venue_ID"], row["Category_Name"], row["Score"]))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations: {e}")

    tk.Label(tab_recommend, text="User ID").pack(pady=5)
    user_id_entry = tk.Entry(tab_recommend, width=30)
    user_id_entry.pack(pady=5)
    tk.Label(tab_recommend, text="Category Name").pack(pady=5)
    category_entry = tk.Entry(tab_recommend, width=30)
    category_entry.pack(pady=5)
    tk.Button(tab_recommend, text="Recommend", command=handle_recommend).pack(pady=10)

    recommendation_output = ttk.Treeview(tab_recommend, columns=("Venue_ID", "Category_Name", "Score"), show="headings")
    recommendation_output.heading("Venue_ID", text="Venue ID")
    recommendation_output.heading("Category_Name", text="Category Name")
    recommendation_output.heading("Score", text="Score")
    recommendation_output.pack(expand=True, fill="both", pady=10)

    # Tab 2: Find Similar Users
    def handle_similar_users():
        user_id = user_id_similar_entry.get()
        if not user_id:
            messagebox.showerror("Error", "Please enter User ID")
            return

        try:
            results = find_top_similar_users(user_id, user_similarity_df, top_n=10)
            if results.empty:
                messagebox.showerror("Error", "No similar users found.")
                return

            similar_users_output.delete(*similar_users_output.get_children())
            for index, score in results.items():
                similar_users_output.insert("", "end", values=(index, score))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find similar users: {e}")

    tk.Label(tab_similar, text="User ID").pack(pady=5)
    user_id_similar_entry = tk.Entry(tab_similar, width=30)
    user_id_similar_entry.pack(pady=5)
    tk.Button(tab_similar, text="Find Similar Users", command=handle_similar_users).pack(pady=10)

    similar_users_output = ttk.Treeview(tab_similar, columns=("User_ID", "Similarity_Score"), show="headings")
    similar_users_output.heading("User_ID", text="User ID")
    similar_users_output.heading("Similarity_Score", text="Similarity Score")
    similar_users_output.pack(expand=True, fill="both", pady=10)

    # Tab 3: Recommend Meeting Place
    def handle_meeting_place():
        user_ids = user_ids_entry.get().split(",")
        if len(user_ids) < 2:
            messagebox.showerror("Error", "Please enter at least 2 User IDs (comma-separated)")
            return

        try:
            _, results = recommend_meeting_place_random_checkins(user_ids, data, k=3)
            if not isinstance(results, pd.DataFrame) or results.empty:
                messagebox.showerror("Error", "No meeting places found.")
                return

            meeting_place_output.delete(*meeting_place_output.get_children())
            for _, row in results.iterrows():
                meeting_place_output.insert("", "end", values=(row["Venue_ID"], row["Category_Name"], row["Latitude"], row["Longitude"], row["Distance_From_Central"]))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find meeting places: {e}")

    tk.Label(tab_meeting, text="User IDs (comma-separated)").pack(pady=5)
    user_ids_entry = tk.Entry(tab_meeting, width=30)
    user_ids_entry.pack(pady=5)
    tk.Button(tab_meeting, text="Recommend Meeting Place", command=handle_meeting_place).pack(pady=10)

    meeting_place_output = ttk.Treeview(tab_meeting, columns=("Venue_ID", "Category_Name", "Latitude", "Longitude", "Distance_From_Central"), show="headings")
    meeting_place_output.heading("Venue_ID", text="Venue ID")
    meeting_place_output.heading("Category_Name", text="Category Name")
    meeting_place_output.heading("Latitude", text="Latitude")
    meeting_place_output.heading("Longitude", text="Longitude")
    meeting_place_output.heading("Distance_From_Central", text="Distance_From_Central")
    meeting_place_output.pack(expand=True, fill="both", pady=10)

    root.mainloop()

if __name__ == "__main__":
    # Step 1: Load Data
    filepath = "data/dataset_NYC.zip"
    print("Loading data...")
    data = load_data(filepath)

    # Step 2: Preprocess and Feature Engineer
    #print("Preprocessing data...")
    data = preprocess_data(data)
    categories_path= "data/categories.zip"
    #print("Feature engineering...")
    data = feature_engineering(data,categories_path)

    # Step 3: Compute User Profiles
    #print("Computing user profiles...")
    user_profiles = compute_user_profile(data)

    # Step 4: Compute User Similarity
    #print("Computing user similarity...")
    user_similarity_df = compute_user_similarity(user_profiles)

    create_gui(data,user_similarity_df)
