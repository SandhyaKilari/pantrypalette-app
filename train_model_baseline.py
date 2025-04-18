import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import sqlite3
import os

# Create model directory if not exists
os.makedirs("../model", exist_ok=True)

# Load cleaned data from SQLite
conn = sqlite3.connect("../database/recipe_data.db")
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# TF-IDF vectorization (1–2 grams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(df["Ingredients"])

# Fit Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
nn_model.fit(X)

# Save model + vectorizer
with open("../model/nn_model.pkl", "wb") as f:
    pickle.dump(nn_model, f)

with open("../model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")