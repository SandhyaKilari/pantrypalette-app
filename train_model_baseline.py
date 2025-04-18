<<<<<<< HEAD:src/train_model_baseline.py
import os
import sqlite3
=======
>>>>>>> 79b0a76 (Added workflow):train_model_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
<<<<<<< HEAD:src/train_model_baseline.py
import joblib

# === Baseline Training Script ===

# Load cleaned recipe data
conn = sqlite3.connect("database/recipe_data.db")
df = pd.read_sql_query("SELECT Title, Ingredients FROM recipes", conn)
conn.close()

# Basic preprocessing: drop nulls
df = df[df["Ingredients"].notnull() & (df["Ingredients"] != "")]

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# TF-IDF vectorizer (simple config)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train = vectorizer.fit_transform(train_df["Ingredients"])

# NearestNeighbors model
model = NearestNeighbors(n_neighbors=10, metric="cosine")
model.fit(X_train)

# Save vectorizer and model
os.makedirs("model", exist_ok=True)
joblib.dump(vectorizer, "model/tfidf_vectorizer_baseline.pkl")
joblib.dump(model, "model/nearest_neighbors_baseline.pkl")

print("Baseline TF-IDF + NearestNeighbors model saved.")
=======
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
>>>>>>> 79b0a76 (Added workflow):train_model_baseline.py
